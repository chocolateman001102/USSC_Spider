#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_scraper.py — Scrape SCOTUS merits briefs AND Oyez oral argument transcripts
for the same dockets, skipping any docket that is missing data in either source.

Pipeline per docket:
  1. Pre-check Oyez  → confirm oral_argument_audio exists
  2. Pre-check SCOTUS → confirm at least one merits brief exists
  3. Download & extract all merits briefs (PDF → JSON) to data/{docket}/
  4. Fetch & save oral argument transcript   to data/{docket}/transcription/

Only dockets that PASS BOTH checks ever get written to disk.
Both crawler.py and oyez_scraper.py are left untouched.

Usage:
    python scripts/combined_scraper.py --term 2023
    python scripts/combined_scraper.py --term 2015 2023
    python scripts/combined_scraper.py --cases 22-915 21-369
    python scripts/combined_scraper.py --all-cases
    python scripts/combined_scraper.py --all-cases --min-year 2000
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# ---------- optional PDF deps ----------
try:
    import fitz  # type: ignore
except Exception:
    fitz = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCOTUS_BASE   = "https://www.supremecourt.gov"
OYEZ_API_BASE = "https://api.oyez.org"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        f.write(data)


def jitter_sleep(base: float, jitter: float = 0.5) -> None:
    time.sleep(base + random.random() * jitter)


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    return filename[:200].strip()


def format_date_compact(date_str: str) -> str:
    """'Jul 20 2021' → '20210720'"""
    try:
        return dt.datetime.strptime(date_str.strip(), "%b %d %Y").strftime("%Y%m%d")
    except Exception:
        return re.sub(r"\s+", "", date_str)[:8]


def extract_party_names(text: str) -> str:
    """Derive a short party name from a brief description string."""
    if not text:
        return "Unknown"
    text = text.strip()
    abbreviations = {
        "Securities and Exchange Commission": "SEC",
        "Federal Trade Commission": "FTC",
        "American Civil Liberties Union": "ACLU",
        "National Association for the Advancement of Colored People": "NAACP",
        "United States": "US",
    }
    patterns = [
        r"(?:reply\s+)?(?:brief|reply)\s+(?:of|for)\s+(?:petitioners?|respondents?)\s+(.+?)(?:\s+filed|\s*$)",
        r"(?:brief|reply)\s+(?:of|for)\s+amici\s+curiae\s+(?:of\s+)?(.+?)(?:\s+filed|\s+in\s+support|\s*$)",
        r"(?:brief|reply)\s+(?:of|for)\s+amicus\s+curiae\s+(?:of\s+)?(.+?)(?:\s+filed|\s+in\s+support|\s*$)",
        r"(?:brief|reply)\s+(?:of|for)\s+(.+?)\s+as\s+amicus\s+curiae",
    ]
    party_name = None
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            party_name = match.group(1).strip()
            party_name = re.sub(r",?\s*et\s+al\.?", "", party_name, flags=re.IGNORECASE)
            party_name = re.sub(r"\s+in\s+support\s+of.+$", "", party_name, flags=re.IGNORECASE)
            party_name = party_name.strip(" .,")
            break
    if not party_name:
        skip = {"brief", "reply", "amici", "amicus", "curiae", "filed", "main", "document",
                "certificate", "word", "count", "proof", "service", "of", "for", "the", "and"}
        words = [w.strip(".,;:") for w in text.split() if w.strip(".,;:").lower() not in skip and len(w.strip(".,;:")) > 1]
        party_name = " ".join(words[:5]) if words else "Unknown"
    for full_name, abbr in abbreviations.items():
        if full_name.lower() in party_name.lower():
            party_name = re.sub(full_name, abbr, party_name, flags=re.IGNORECASE)
    party_name = re.sub(r",?\s*et\s+al\.?", "", party_name, flags=re.IGNORECASE)
    party_name = re.sub(r"\s*,\s*", " ", party_name)
    party_name = re.sub(r"\s+", " ", party_name).strip()
    words = [w for w in party_name.split() if len(w) > 2 or w.isupper()]
    party_name = " ".join(words[:6]) if words else party_name
    party_name = sanitize_filename(party_name)
    return (party_name[:70].strip() if len(party_name) > 70 else party_name) or "Unknown"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NotFoundError(Exception):
    pass

class SiteBlockedError(Exception):
    pass

class UnexpectedContentType(Exception):
    pass


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class Downloader:
    def __init__(self, session: requests.Session, user_agent: Optional[str] = None, min_interval: float = 1.0):
        self.session = session
        self.min_interval = min_interval
        if user_agent:
            self.session.headers.update({"User-Agent": user_agent})

    def get(self, url: str, *, referer: Optional[str] = None, stream: bool = False, max_attempts: int = 3) -> requests.Response:
        headers: Dict[str, str] = {}
        if referer:
            headers["Referer"] = referer
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.get(url, headers=headers, timeout=30, allow_redirects=True, stream=stream)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise requests.RequestException(f"HTTP {resp.status_code}")
                return resp
            except requests.RequestException as e:
                last_exc = e
                if attempt == max_attempts:
                    raise
                jitter_sleep(min(30, 2 ** (attempt - 1)))
        raise last_exc or requests.RequestException("request failed")

    def download_pdf(self, url: str, *, referer: Optional[str] = None, max_size_mb: int = 100) -> Tuple[bytes, Dict[str, Any]]:
        resp = self.get(url, referer=referer, stream=True)
        ctype = resp.headers.get("Content-Type") or ""
        if "pdf" not in ctype.lower():
            logging.warning("Unexpected Content-Type: %s", ctype)
        total = 0
        chunks: List[bytes] = []
        limit = max_size_mb * 1024 * 1024
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total > limit:
                    resp.close()
                    raise UnexpectedContentType(f"File too large: > {max_size_mb} MB")
        data = b"".join(chunks)
        meta = {"status_code": resp.status_code, "headers": dict(resp.headers), "content_type": ctype, "size_bytes": total}
        jitter_sleep(self.min_interval)
        return data, meta


# ---------------------------------------------------------------------------
# PDF Extractor
# ---------------------------------------------------------------------------

class PDFExtractor:
    def __init__(self, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr

    def extract_text_pdf(self, pdf_bytes: bytes) -> Tuple[List[str], str, str]:
        if pdfminer_extract_text is not None:
            try:
                text = pdfminer_extract_text(io.BytesIO(pdf_bytes)) or ""
                pages = [p.strip() for p in re.split(r"\f|\n\s*\n\s*\n", text) if p.strip()]
                return pages, text, "pdfminer"
            except Exception as e:
                logging.exception("pdfminer extract failed: %s", e)
        if fitz is not None:
            try:
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:  # type: ignore
                    pg_texts = [page.get_text() for page in doc]
                pages = [p.strip() for p in pg_texts if p and p.strip()]
                return pages, "\n\n".join(pg_texts), "pymupdf"
            except Exception as e:
                logging.exception("pymupdf extract failed: %s", e)
        return [], "", "none"

    def extract(self, pdf_bytes: bytes) -> Dict[str, Any]:
        pages, full, engine = self.extract_text_pdf(pdf_bytes)
        return {"pages": pages, "full": full, "method": "text", "notes": engine}


# ---------------------------------------------------------------------------
# SCOTUS docket adapter
# ---------------------------------------------------------------------------

@dataclass
class ResolvedDoc:
    query_code: str
    source_page_url: str
    download_url: str
    title: Optional[str] = None
    date: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def _normalize_scotus_code(raw: str) -> Tuple[str, List[str]]:
    s = (raw or "").strip().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^(.*?)(A)([0-9]+)$", s)
    if m:
        s = m.group(1) + "a" + m.group(3)
    primary = s
    alts: List[str] = []
    up = primary.upper()
    if up != primary:
        alts.append(up)
    return primary, alts


class ScotusDocketAdapter:
    def __init__(self, session: requests.Session, base_url: str = SCOTUS_BASE):
        self.session = session
        self.base_url = base_url.rstrip("/")

    def _build_url(self, code: str) -> str:
        return f"{self.base_url}/search.aspx?filename=/docketfiles/{code}.htm"

    def _build_alt_url(self, code: str) -> str:
        return f"{self.base_url}/docket/docketfiles/html/public/{code}.html"

    def _is_blank_results(self, soup: BeautifulSoup) -> bool:
        text = soup.get_text(" ", strip=True)
        return ("Search Results" in text or "Search" in text) and "No." not in text

    def _iter_pdf_links(self, soup: BeautifulSoup):
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            low = href.lower()
            if not (low.endswith(".pdf") or "/docketpdf/" in low):
                continue
            desc = a.get_text(" ", strip=True)
            if desc.strip().lower() != "main document":
                continue
            parent_text = desc
            date_text = ""
            row_text = ""
            current = a.find_parent()
            while current:
                if current.name == "td":
                    parent_text = current.get_text(" ", strip=True)
                    date_cell = current.find_previous_sibling("td", class_="ProceedingDate")
                    if date_cell:
                        date_text = date_cell.get_text(" ", strip=True)
                    row = current.find_parent("tr")
                    if row:
                        row_text = row.get_text(" ", strip=True)
                    break
                current = current.find_parent()
            yield href, desc, parent_text, date_text, row_text

    def collect_briefs(self, query_code: str) -> List[ResolvedDoc]:
        """Return all merits-stage brief ResolvedDocs for this docket, or [] if none found."""
        primary, alts = _normalize_scotus_code(query_code)
        candidates = [primary] + alts
        docs: List[ResolvedDoc] = []
        found_valid_page = False

        BRIEF_PREFIXES = (
            "brief of petitioner", "brief for petitioner",
            "brief of respondent", "brief for respondent",
            "reply brief", "reply of petitioner", "reply for petitioner",
            "reply of respondent", "reply for respondent",
        )

        for cand in candidates:
            if found_valid_page:
                break
            for url in (self._build_alt_url(cand), self._build_url(cand)):
                r = self.session.get(url, timeout=30, allow_redirects=True)
                if r.status_code != 200:
                    continue
                final_url = r.url or url
                low = final_url.lower()
                if not (low.endswith(f"/{cand.lower()}.htm") or low.endswith(f"/{cand.lower()}.html")
                        or low.endswith(f"{cand.lower()}.htm") or low.endswith(f"{cand.lower()}.html")):
                    continue
                soup = BeautifulSoup(r.text, "lxml")
                if self._is_blank_results(soup):
                    continue
                found_valid_page = True

                for href, desc, parent_text, date_text, row_text in self._iter_pdf_links(soup):
                    row_label = row_text.lower()[:120]
                    if not any(p in row_label for p in BRIEF_PREFIXES):
                        continue
                    full_url = href if href.startswith("http") else f"{self.base_url}{href if href.startswith('/') else '/' + href}"

                    # role
                    rl = row_text.lower()[:80]
                    if "reply" in rl:
                        role = "reply"
                    elif "petitioner" in rl:
                        role = "petitioner"
                    elif "respondent" in rl:
                        role = "respondent"
                    else:
                        role = "brief"

                    date_part = format_date_compact(date_text) if date_text else ""
                    party_name = extract_party_names(parent_text)
                    parts = [p for p in [date_part, role, party_name] if p]
                    custom_filename = sanitize_filename("_".join(parts).replace(" ", "_"))[:90]

                    docs.append(ResolvedDoc(
                        query_code=query_code,
                        source_page_url=final_url,
                        download_url=full_url,
                        title=desc or parent_text,
                        date=date_text or None,
                        extra={"date": date_text, "custom_filename": custom_filename},
                    ))
                break  # stop trying URL variants once a valid page is found

        logging.info("Found %d brief doc(s) for %s on SCOTUS", len(docs), query_code)
        return docs


# ---------------------------------------------------------------------------
# Oyez helpers
# ---------------------------------------------------------------------------

def docket_to_terms(docket: str) -> List[str]:
    m = re.match(r"^(\d{2})-", docket)
    if not m:
        return []
    yy = int(m.group(1))
    current_yy = int(dt.datetime.now(dt.timezone.utc).strftime("%y"))
    if yy <= current_yy:
        base, alt = 2000 + yy, 1900 + yy
    else:
        base, alt = 1900 + yy, 2000 + yy
    seen: set = set()
    result = []
    for y in [base, base + 1, base - 1, alt]:
        s = str(y)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def fetch_case_meta(session: requests.Session, docket: str, min_interval: float) -> Optional[Dict]:
    """Try all plausible terms for a docket. Returns parsed JSON or None."""
    for term in docket_to_terms(docket):
        url = f"{OYEZ_API_BASE}/cases/{term}/{docket}"
        try:
            resp = session.get(url, timeout=20)
            jitter_sleep(min_interval)
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                logging.warning("HTTP %d at %s", resp.status_code, url)
                continue
            data = resp.json()
            if isinstance(data, dict) and data.get("docket_number"):
                logging.info("Found case %s in term %s on Oyez", docket, term)
                return data
        except Exception as e:
            logging.warning("Error fetching %s: %s", url, e)
    logging.info("No case found on Oyez for %s", docket)
    return None


def fetch_oral_argument_transcript(session: requests.Session, audio_href: str, min_interval: float) -> Optional[str]:
    try:
        resp = session.get(audio_href, timeout=30)
        jitter_sleep(min_interval)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception as e:
        logging.warning("Error fetching transcript %s: %s", audio_href, e)
        return None

    sections = (data.get("transcript") or {}).get("sections") or []
    lines: List[str] = []
    for section in sections:
        for turn in (section.get("turns") or []):
            speaker_name = (turn.get("speaker") or {}).get("name") or "Unknown"
            turn_text = " ".join(
                (b.get("text") or "").strip()
                for b in (turn.get("text_blocks") or [])
                if b.get("text")
            ).strip()
            if turn_text:
                lines.append(f"{speaker_name}: {turn_text}")
    return "\n".join(lines) if lines else None


def build_meta_from_case(case: Dict) -> Dict:
    meta: Dict[str, Any] = {"win_side": None, "votes_side": {}, "advocates": {}, "conversation_ids": []}
    decisions = case.get("decisions") or []
    if decisions:
        d = decisions[0]
        if d.get("winning_party") == case.get("first_party"):
            meta["win_side"] = 1
        elif d.get("winning_party") == case.get("second_party"):
            meta["win_side"] = 0
        for v in (d.get("votes") or []):
            member = v.get("member") or {}
            identifier = member.get("identifier") or ""
            vote = v.get("vote") or ""
            if meta["win_side"] == 1:
                meta["votes_side"][f"j__{identifier}"] = 1 if vote == "majority" else 0
            else:
                meta["votes_side"][f"j__{identifier}"] = 0 if vote == "majority" else 1
    for a in (case.get("advocates") or []):
        adv = a.get("advocate") or {}
        identifier = adv.get("identifier") or adv.get("name") or ""
        description = (a.get("advocate_description") or "").strip()
        desc_lower = description.lower()
        side = 1 if "petitioner" in desc_lower else (0 if "respondent" in desc_lower else None)
        meta["advocates"][identifier] = {"side": side, "role": description}
    return meta


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _make_path(output_dir: Path, query_code: str, sha256_hex: str, suffix: str, custom_name: Optional[str] = None) -> Path:
    subdir = "pdf" if suffix == ".pdf" else "json"
    case_dir = output_dir / query_code / subdir
    name = f"{custom_name}_{sha256_hex[:6]}{suffix}" if custom_name else f"{query_code}_{sha256_hex[:8]}{suffix}"
    return case_dir / name


def write_pdf(output_dir: Path, query_code: str, sha: str, data: bytes, custom_name: Optional[str] = None) -> Path:
    path = _make_path(output_dir, query_code, sha, ".pdf", custom_name)
    save_bytes(path, data)
    return path


def write_json(output_dir: Path, query_code: str, sha: str, payload: Dict, custom_name: Optional[str] = None) -> Path:
    path = _make_path(output_dir, query_code, sha, ".json", custom_name)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# Oyez case-discovery (shared with crawler logic)
# ---------------------------------------------------------------------------

def discover_cases_from_oyez(session: requests.Session, term: Optional[int] = None,
                              page_size: int = 100, min_interval: float = 1.0):
    """Generator: yield docket numbers (XX-NNNN format) from the Oyez API."""
    base = f"{OYEZ_API_BASE}/cases"
    page = 0
    seen: set = set()
    docket_re = re.compile(r"^\d{2}-\d+$")
    while True:
        params: dict = {"per_page": page_size, "page": page}
        if term is not None:
            params["filter"] = f"term:{term}"
        try:
            r = session.get(base, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logging.error("Oyez API error on page %d: %s", page, e)
            break
        if not data:
            break
        found_any = False
        for case in data:
            docket = str(case.get("docket_number") or "").strip()
            if not docket or not docket_re.match(docket):
                continue
            if re.match(r"^\d{2}A\d+$", docket, re.IGNORECASE):
                continue
            if docket in seen:
                continue
            seen.add(docket)
            found_any = True
            yield docket
        logging.info("Oyez page %d: fetched %d cases, %d unique dockets so far", page, len(data), len(seen))
        page += 1
        time.sleep(min_interval)
        if not found_any and len(data) < page_size:
            break


# ---------------------------------------------------------------------------
# Combined per-docket processor
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    total: int = 0
    both_found: int = 0
    skipped_no_transcript: int = 0
    skipped_no_briefs: int = 0
    failed: int = 0


def process_one_combined(
    docket: str,
    scotus_adapter: ScotusDocketAdapter,
    downloader: Downloader,
    extractor: PDFExtractor,
    oyez_session: requests.Session,
    output_dir: Path,
    min_interval: float,
    overwrite: bool,
) -> Tuple[str, str]:
    """
    Returns (status, message) where status is one of:
      'success', 'skipped_no_transcript', 'skipped_no_briefs', 'failed'
    """
    try:
        # ---- Step 1: Pre-check Oyez (fast — no transcript fetch yet) ----
        case_meta = fetch_case_meta(oyez_session, docket, min_interval)
        if not case_meta:
            return "skipped_no_transcript", "docket not found on Oyez"
        audio_entries = case_meta.get("oral_argument_audio") or []
        if not audio_entries:
            logging.info("[%s] No oral argument audio on Oyez — skipping", docket)
            return "skipped_no_transcript", "no oral_argument_audio on Oyez"

        # ---- Step 2: Pre-check SCOTUS ----
        brief_docs = scotus_adapter.collect_briefs(docket)
        if not brief_docs:
            logging.info("[%s] No merits briefs on SCOTUS — skipping", docket)
            return "skipped_no_briefs", "no merits briefs found on SCOTUS"

        logging.info("[%s] Both sources confirmed — downloading %d brief(s) + transcript", docket, len(brief_docs))

        # ---- Step 3: Check overwrite for transcript ----
        transcript_path = output_dir / docket / "transcription" / f"{docket}__corpus.json"
        if transcript_path.exists() and not overwrite:
            # Check briefs too — if all json files already exist, skip fully
            existing_jsons = list((output_dir / docket / "json").glob("*.json")) if (output_dir / docket / "json").exists() else []
            if existing_jsons:
                logging.info("[%s] All outputs already exist (use --overwrite). Skipping.", docket)
                return "skipped_no_transcript", "already exists (use --overwrite)"

        # ---- Step 4: Download & extract briefs ----
        for i, resolved in enumerate(brief_docs, 1):
            custom_name = resolved.extra.get("custom_filename") if resolved.extra else None

            # Skip if already downloaded
            if not overwrite:
                existing = list((output_dir / docket / "json").glob(f"{custom_name}_*.json")) if custom_name and (output_dir / docket / "json").exists() else []
                if existing:
                    logging.info("[%s] Brief %d/%d already exists, skipping download", docket, i, len(brief_docs))
                    continue

            try:
                pdf_bytes, _ = downloader.download_pdf(resolved.download_url, referer=resolved.source_page_url)
            except Exception as e:
                logging.warning("[%s] Failed to download brief %d/%d: %s", docket, i, len(brief_docs), e)
                continue

            sha = sha256_bytes(pdf_bytes)
            pdf_path = write_pdf(output_dir, docket, sha, pdf_bytes, custom_name)

            pages_count = 0
            if fitz is not None:
                try:
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:  # type: ignore
                        pages_count = doc.page_count
                except Exception:
                    pass

            extr = extractor.extract(pdf_bytes)
            payload = {
                "version": "1.0",
                "document": {
                    "query_code": docket,
                    "source_page_url": resolved.source_page_url,
                    "download_url": resolved.download_url,
                    "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
                    "sha256": sha,
                    "filename": pdf_path.name,
                    "file_size_bytes": len(pdf_bytes),
                    "pages": pages_count,
                    "extraction": {"method": extr["method"], "chars": len(extr["full"] or ""), "notes": extr["notes"]},
                    "metadata": {
                        "title": resolved.title,
                        "date": resolved.extra.get("date") if resolved.extra else None,
                        "extra": resolved.extra or {},
                    },
                    "content": {"page_text": extr["pages"], "full_text": extr["full"]},
                },
            }
            json_path = write_json(output_dir, docket, sha, payload, custom_name)
            logging.info("[%s] Brief %d/%d saved → %s", docket, i, len(brief_docs), json_path)

        # ---- Step 5: Fetch & save transcript ----
        if transcript_path.exists() and not overwrite:
            logging.info("[%s] Transcript already exists — skipping Oyez fetch", docket)
        else:
            all_parts: List[str] = []
            for entry in audio_entries:
                href = entry.get("href")
                if not href:
                    continue
                text = fetch_oral_argument_transcript(oyez_session, href, min_interval)
                if text:
                    all_parts.append(text)

            if not all_parts:
                logging.warning("[%s] Could not extract any transcript text from Oyez", docket)
                # briefs were already saved; report partial success
                return "success", "briefs saved; transcript text empty"

            full_transcript = "\n\n".join(all_parts)
            meta = build_meta_from_case(case_meta)
            transcript_payload = {
                "docket": docket,
                "utterence": full_transcript,
                "meta": meta,
                "source": {
                    "corpus": "oyez",
                    "docs": "https://www.oyez.org",
                    "api": f"{OYEZ_API_BASE}/cases/{case_meta.get('term', 'unknown')}/{docket}",
                },
            }
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with transcript_path.open("w", encoding="utf-8") as f:
                json.dump(transcript_payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
            logging.info("[%s] Transcript saved → %s (%d chars)", docket, transcript_path, len(full_transcript))

        return "success", f"{len(brief_docs)} brief(s) + transcript"

    except Exception as e:
        logging.exception("[%s] FAILED: %s", docket, e)
        return "failed", str(e)


# ---------------------------------------------------------------------------
# Logger init
# ---------------------------------------------------------------------------

def init_logger(output_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "output" / "logs"
    ensure_dir(log_dir)
    log_path = log_dir / "combined_scraper.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined SCOTUS brief + Oyez transcript scraper. "
                    "Only processes dockets with data available in BOTH databases."
    )
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Specific docket numbers to process, e.g. 22-915 21-432")
    ap.add_argument("--all-cases", action="store_true",
                    help="Auto-discover ALL cases via the Oyez API")
    ap.add_argument("--term", type=int, nargs="+", default=None, metavar="YEAR",
                    help="One year or a range, e.g. --term 2023 or --term 2015 2023")
    ap.add_argument("--output-dir", type=Path, default=Path("./data"),
                    help="Root directory for all output (default: ./data)")
    ap.add_argument("--min-interval", type=float, default=1.5,
                    help="Seconds between requests (default: 1.5)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download and overwrite existing files")
    args = ap.parse_args()

    if not args.all_cases and args.term is None and not args.cases:
        ap.error("Provide one of: --cases DOCKET..., --all-cases, or --term YEAR [YEAR]")

    ensure_dir(args.output_dir)
    init_logger(args.output_dir)

    # Build sessions (separate so SCOTUS & Oyez each see the right User-Agent)
    scotus_session = requests.Session()
    scotus_session.headers.update({
        "User-Agent": "combined-scraper/1.0",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    })
    oyez_session = requests.Session()
    oyez_session.headers.update({
        "User-Agent": "USSC-Spider/1.0 (academic research; contact via GitHub)",
        "Accept": "application/json",
    })

    scotus_adapter = ScotusDocketAdapter(scotus_session, SCOTUS_BASE)
    downloader = Downloader(scotus_session, min_interval=args.min_interval)
    extractor = PDFExtractor(enable_ocr=False)

    # ---- docket filter ----
    def _filter(code: str) -> bool:
        if not re.match(r"^\d{2}-\d+$", code):
            return False
        if re.match(r"^\d{2}A\d+$", code, re.IGNORECASE):
            return False
        return True

    # ---- build task list ----
    if args.cases:
        dockets = [d for d in args.cases if _filter(d)]
    elif args.term or args.all_cases:
        term_years: List[int] = []
        if args.term:
            if len(args.term) > 2:
                ap.error("--term accepts 1 or 2 year values")
            start, end = sorted([args.term[0], args.term[-1]])
            term_years = list(range(start, end + 1))
            logging.info("Discovering cases for term(s) %s via Oyez API…", term_years)
        else:
            logging.info("Discovering ALL cases via Oyez API…")

        seen_global: set = set()
        dockets = []
        if term_years:
            for yr in term_years:
                for code in discover_cases_from_oyez(oyez_session, term=yr, min_interval=args.min_interval):
                    if code not in seen_global and _filter(code):
                        seen_global.add(code)
                        dockets.append(code)
        else:
            for code in discover_cases_from_oyez(oyez_session, term=None, min_interval=args.min_interval):
                if _filter(code) and code not in seen_global:
                    seen_global.add(code)
                    dockets.append(code)

        logging.info("Discovered %d dockets total", len(dockets))
    else:
        dockets = []

    if not dockets:
        logging.warning("No dockets to process.")
        return

    # ---- run ----
    stats = RunStats()
    for i, docket in enumerate(dockets, 1):
        logging.info("=" * 60)
        logging.info("[%d/%d] Processing: %s", i, len(dockets), docket)
        stats.total += 1
        status, msg = process_one_combined(
            docket=docket,
            scotus_adapter=scotus_adapter,
            downloader=downloader,
            extractor=extractor,
            oyez_session=oyez_session,
            output_dir=args.output_dir,
            min_interval=args.min_interval,
            overwrite=args.overwrite,
        )
        if status == "success":
            stats.both_found += 1
            logging.info("[%s] SUCCESS: %s", docket, msg)
        elif status == "skipped_no_transcript":
            stats.skipped_no_transcript += 1
            logging.info("[%s] SKIP (no transcript): %s", docket, msg)
        elif status == "skipped_no_briefs":
            stats.skipped_no_briefs += 1
            logging.info("[%s] SKIP (no briefs): %s", docket, msg)
        else:
            stats.failed += 1
            logging.error("[%s] FAILED: %s", docket, msg)

    logging.info("=" * 60)
    logging.info(
        "DONE  total=%-4d  both_found=%-4d  skip_no_transcript=%-4d  skip_no_briefs=%-4d  failed=%-4d",
        stats.total, stats.both_found, stats.skipped_no_transcript, stats.skipped_no_briefs, stats.failed,
    )


if __name__ == "__main__":
    main()
