#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oyez_scraper.py — Fetch oral argument transcripts from the Oyez public API.

Writes {docket}__corpus.json files compatible with process_similarity.py.
No API key or account needed — api.oyez.org is a fully public API.

API flow:
  1. GET https://api.oyez.org/cases/{term}/{docket}
     → locate oral_argument_audio[].href
  2. GET {oral_argument_audio_href}
     → parse transcript.sections[].turns[] into speaker-tagged text
  3. Save data/{docket}/json/{docket}__corpus.json

Usage:
    # Specific cases
    python scripts/oyez_scraper.py --cases 22-915 21-432

    # All case folders under data/ that are missing a corpus JSON
    python scripts/oyez_scraper.py --all-cases

    # Force overwrite existing corpus files
    python scripts/oyez_scraper.py --all-cases --overwrite

    # Fetch and process all cases for a specific term from the Oyez API
    python scripts/oyez_scraper.py --term 2023
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OYEZ_API_BASE = "https://api.oyez.org"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MIN_INTERVAL = 1.5  # seconds between API requests (be polite)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def docket_to_terms(docket: str) -> List[str]:
    """
    Infer possible SCOTUS terms from a docket number's two-digit year prefix.

    Oyez stores cases under the October Term (OT) year. OT20XX runs Oct 20XX –
    Jul 20XX+1, so a docket like "22-915" (granted OT2022) can appear at either
    cases/2022/ or cases/2023/ depending on when Oyez indexed it.
    We therefore try: 20YY, 20YY+1, 20YY-1, and 19YY as fallback.

    e.g. "22-915"  → ["2022", "2023", "2021", "1922"]
         "17-773"  → ["2017", "2018", "2016", "1917"]
    """
    m = re.match(r"^(\d{2})-", docket)
    if not m:
        return []
    yy = int(m.group(1))
    import datetime
    current_yy = int(datetime.datetime.now(datetime.timezone.utc).strftime("%y"))
    if yy <= current_yy:
        base = 2000 + yy
        alt = 1900 + yy
    else:
        base = 1900 + yy
        alt = 2000 + yy
    # Deduplicated, ordered: base, base+1, base-1, alt-century fallback
    seen: set = set()
    result = []
    for y in [base, base + 1, base - 1, alt]:
        s = str(y)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def jitter_sleep(base: float) -> None:
    import random
    time.sleep(base + random.random() * 0.5)


# ---------------------------------------------------------------------------
# Oyez API calls
# ---------------------------------------------------------------------------

def fetch_case_meta(session: requests.Session, docket: str, min_interval: float) -> Optional[Dict]:
    """
    Try all plausible terms for a docket. Returns parsed JSON or None.
    """
    terms = docket_to_terms(docket)
    if not terms:
        log.warning("Cannot infer term from docket: %s", docket)
        return None

    for term in terms:
        url = f"{OYEZ_API_BASE}/cases/{term}/{docket}"
        log.info("Trying: %s", url)
        try:
            resp = session.get(url, timeout=20)
            jitter_sleep(min_interval)
            if resp.status_code == 404:
                log.info("404 Not Found: %s", url)
                continue
            if resp.status_code != 200:
                log.warning("HTTP %d at %s", resp.status_code, url)
                continue
            data = resp.json()
            # Sanity-check: should be a dict with a docket_number field
            if isinstance(data, dict) and data.get("docket_number"):
                log.info("Found case %s in term %s", docket, term)
                return data
        except Exception as e:
            log.warning("Error fetching %s: %s", url, e)
            continue

    log.warning("Could not find case %s on Oyez (tried terms: %s)", docket, terms)
    return None


def fetch_oral_argument_transcript(
    session: requests.Session, audio_href: str, min_interval: float
) -> Optional[str]:
    """
    Fetch a single oral argument audio resource and extract the transcript
    as a plain text string in ConvoKit format: "Speaker Name: text\n..."
    """
    log.info("Fetching oral argument transcript: %s", audio_href)
    try:
        resp = session.get(audio_href, timeout=30)
        jitter_sleep(min_interval)
        if resp.status_code != 200:
            log.warning("HTTP %d for %s", resp.status_code, audio_href)
            return None
        data = resp.json()
    except Exception as e:
        log.warning("Error fetching transcript %s: %s", audio_href, e)
        return None

    transcript = data.get("transcript") or {}
    sections = transcript.get("sections") or []

    lines: List[str] = []
    for section in sections:
        turns = section.get("turns") or []
        for turn in turns:
            speaker_info = turn.get("speaker") or {}
            speaker_name = speaker_info.get("name") or "Unknown"
            text_blocks = turn.get("text_blocks") or []
            # Concatenate all text_blocks in this turn
            turn_text = " ".join(
                (b.get("text") or "").strip()
                for b in text_blocks
                if b.get("text")
            ).strip()
            if turn_text:
                lines.append(f"{speaker_name}: {turn_text}")

    if not lines:
        log.warning("Empty transcript at %s", audio_href)
        return None

    return "\n".join(lines)


def build_meta_from_case(case: Dict) -> Dict:
    """
    Extract win_side, votes_side, and advocates from Oyez case JSON.
    Mirrors the ConvoKit schema used by generate_docket_corpus_json.py.
    """
    meta: Dict[str, Any] = {
        "win_side": None,
        "votes_side": {},
        "advocates": {},
        "conversation_ids": [],
    }

    decisions = case.get("decisions") or []
    if decisions:
        d = decisions[0]
        # Winning party side — Oyez doesn't use a numeric side directly on the decision,
        # but we can derive it: "Petitioner" wins → 1, "Respondent" wins → 0
        winning = case.get("first_party_label", "")  if d.get("winning_party") == case.get("first_party") else ""
        if d.get("winning_party") == case.get("first_party"):
            meta["win_side"] = 1
        elif d.get("winning_party") == case.get("second_party"):
            meta["win_side"] = 0

        votes = d.get("votes") or []
        for v in votes:
            member = v.get("member") or {}
            identifier = member.get("identifier") or ""
            name = member.get("name") or ""
            vote = v.get("vote") or ""
            # Map "majority" → 1, "minority" → 0 (petitioner wins = 1)
            if meta["win_side"] == 1:
                meta["votes_side"][f"j__{identifier}"] = 1 if vote == "majority" else 0
            else:
                meta["votes_side"][f"j__{identifier}"] = 0 if vote == "majority" else 1

    advocates = case.get("advocates") or []
    for a in advocates:
        adv = a.get("advocate") or {}
        identifier = adv.get("identifier") or adv.get("name") or ""
        description = (a.get("advocate_description") or "").strip()
        # Infer side
        side = None
        desc_lower = description.lower()
        if "petitioner" in desc_lower:
            side = 1
        elif "respondent" in desc_lower:
            side = 0
        meta["advocates"][identifier] = {
            "side": side,
            "role": description,
        }

    return meta


def scrape_case(
    session: requests.Session,
    docket: str,
    data_dir: Path,
    min_interval: float,
    overwrite: bool,
    skip_existing: bool,
) -> str:
    """
    Main per-case logic. Returns status string: 'success', 'skipped', 'not_found', 'failed'.
    """
    out_path = data_dir / docket / "transcription" / f"{docket}__corpus.json"

    if out_path.exists() and not overwrite:
        if skip_existing:
            log.info("SKIP %s — corpus file already exists. Use --overwrite to replace.", docket)
            return "skipped"

    # 1. Fetch case metadata
    case = fetch_case_meta(session, docket, min_interval)
    if not case:
        return "not_found"

    # 2. Fetch oral argument transcript(s)
    audio_entries = case.get("oral_argument_audio") or []
    if not audio_entries:
        log.warning("No oral argument audio found for %s", docket)
        return "not_found"

    all_transcript_parts: List[str] = []
    for entry in audio_entries:
        href = entry.get("href")
        if not href:
            continue
        text = fetch_oral_argument_transcript(session, href, min_interval)
        if text:
            all_transcript_parts.append(text)

    if not all_transcript_parts:
        log.warning("Could not extract any transcript text for %s", docket)
        return "not_found"

    full_transcript = "\n\n".join(all_transcript_parts)

    # 3. Build meta
    meta = build_meta_from_case(case)

    # 4. Assemble payload (same schema as generate_docket_corpus_json.py)
    payload = {
        "docket": docket,
        "utterence": full_transcript,   # keep the same (intentional) spelling as ConvoKit
        "meta": meta,
        "source": {
            "corpus": "oyez",
            "docs": "https://www.oyez.org",
            "api": f"{OYEZ_API_BASE}/cases/{case.get('term', 'unknown')}/{docket}",
        },
    }

    # 5. Write to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    char_count = len(full_transcript)
    log.info("SUCCESS %s → %s (%d chars)", docket, out_path, char_count)
    return "success"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch oral argument transcripts from Oyez and save as __corpus.json files."
    )
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help="Docket numbers to process, e.g. 22-915 21-432"
    )
    parser.add_argument(
        "--all-cases", action="store_true",
        help="Process all case directories under --data-dir"
    )
    parser.add_argument(
        "--term", type=int, nargs='+', default=None, metavar='YEAR',
        help="One year (--term 2023) or a range (--term 1997 2003)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Root data directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--min-interval", type=float, default=DEFAULT_MIN_INTERVAL,
        help="Minimum seconds between API requests (default: 1.5)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing __corpus.json files"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip dockets that already have a __corpus.json (default: True)"
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir

    session = requests.Session()
    session.headers.update({
        "User-Agent": "USSC-Spider/1.0 (academic research; contact via GitHub)",
        "Accept": "application/json",
    })

    # Resolve list of dockets to process
    if args.term:
        if len(args.term) > 2:
            parser.error("--term accepts 1 or 2 year values")
        start_yr, end_yr = (args.term[0], args.term[-1])
        if start_yr > end_yr:
            start_yr, end_yr = end_yr, start_yr
        years = list(range(start_yr, end_yr + 1))
        log.info("Fetching cases for term(s) %s from Oyez API", years)
        docket_re = re.compile(r"^\d{2}-\d+$")
        seen: set = set()
        dockets = []
        for yr in years:
            url = f"{OYEZ_API_BASE}/cases?per_page=200&filter=term:{yr}"
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                for c in resp.json():
                    d = str(c.get("docket_number") or "").strip()
                    if d and docket_re.match(d) and d not in seen:
                        seen.add(d)
                        dockets.append(d)
            else:
                log.error("Failed to fetch term %d: HTTP %d", yr, resp.status_code)
            jitter_sleep(args.min_interval)
        log.info("Found %d cases across term(s) %s", len(dockets), years)
    elif args.all_cases:
        log.info("Fetching ALL cases from Oyez API (paginating through all terms)…")
        dockets = []
        page = 0
        page_size = 100
        docket_re = re.compile(r"^\d{2}-\d+$")
        seen: set = set()
        while True:
            url = f"{OYEZ_API_BASE}/cases?per_page={page_size}&page={page}"
            log.info("Fetching Oyez page %d…", page)
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                log.error("Oyez API error on page %d: HTTP %d", page, resp.status_code)
                break
            data = resp.json()
            if not data:
                break
            added = 0
            for c in data:
                d = str(c.get("docket_number") or "").strip()
                if d and docket_re.match(d) and d not in seen:
                    seen.add(d)
                    dockets.append(d)
                    added += 1
            log.info("Page %d: %d new dockets (total so far: %d)", page, added, len(dockets))
            page += 1
            jitter_sleep(args.min_interval)
            if len(data) < page_size:
                break  # last page
        log.info("Discovered %d dockets from Oyez API", len(dockets))
    elif args.cases:
        dockets = args.cases
    else:
        parser.error("Specify --cases DOCKET..., --all-cases, or --term TERM")
        return

    if not dockets:
        log.warning("No dockets to process.")
        return

    counts = {"success": 0, "skipped": 0, "not_found": 0, "failed": 0}

    for i, docket in enumerate(dockets, 1):
        log.info("[%d/%d] Processing: %s", i, len(dockets), docket)
        try:
            status = scrape_case(
                session=session,
                docket=docket,
                data_dir=data_dir,
                min_interval=args.min_interval,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            log.exception("FAILED %s: %s", docket, e)
            status = "failed"
        counts[status] = counts.get(status, 0) + 1

    log.info(
        "DONE: total=%d success=%d skipped=%d not_found=%d failed=%d",
        len(dockets), counts["success"], counts["skipped"],
        counts["not_found"], counts["failed"],
    )


if __name__ == "__main__":
    main()
