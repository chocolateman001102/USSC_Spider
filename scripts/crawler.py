#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crawler (JSON only, SCOTUS)：读取 JSON/JSONL -> 命中 SCOTUS docket 页面 -> 下载 PDF -> 提取文本 -> 输出 JSON

本版特性
- 仅支持 JSON 或 JSONL 输入（必须含字段 docket_no）；已移除 CSV 读取逻辑
- 只处理正式案号“YY-数字”（如 75-1552）；忽略 A 号（如 24A949）
- SCOTUS 直链解析（无需在搜索页提交表单）
- 文本抽取优先 pdfminer.six；若缺失则回退 PyMuPDF；必要时可启用 OCR 兜底

用法


依赖（最小集）
    pip install requests beautifulsoup4 lxml pymupdf
    # 可选：更高质量文本抽取
    pip install pdfminer.six
    # 可选 OCR

输出目录
    {output}/pdf/YYYY/MM/{query}_{hash8}.pdf
    {output}/json/YYYY/MM/{query}_{hash8}.json
    {output}/logs/app.log
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
from typing import List, Optional, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup

# ---------- 可选依赖：PyMuPDF ----------
try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

# ---------- 可选依赖：pdfminer.six ----------
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:  # pragma: no cover
    pdfminer_extract_text = None  # type: ignore

JSON_SCHEMA: Dict[str, Any] = {
    "version": "1.0",
    "document": {
        "query_code": "string",
        "source_page_url": "string",
        "download_url": "string",
        "fetched_at": "ISO8601",
        "sha256": "hex",
        "filename": "string",
        "file_size_bytes": 0,
        "pages": 0,
        "extraction": {
            "method": "text|ocr|none",
            "chars": 0,
            "notes": "string"
        },
        "metadata": {
            "title": "string",
            "date": "string",
            "extra": {}
        },
        "content": {
            "page_text": ["string", "..."],
            "full_text": "string"
        }
    }
}

# ----------------------------- 数据类与异常 -----------------------------

@dataclass
class ResolvedDoc:
    query_code: str
    source_page_url: str
    download_url: str
    title: Optional[str] = None
    date: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class NotFoundError(Exception):
    pass

class SiteBlockedError(Exception):
    pass

class UnexpectedContentType(Exception):
    pass

# ----------------------------- 工具函数 -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    with open(path, 'wb') as f:
        f.write(data)


def make_output_paths(output_dir: Path, query_code: str, sha256_hex: str, suffix: str, custom_name: Optional[str] = None) -> Path:
    """创建输出路径。Custom names always include a short hash suffix to guarantee uniqueness."""
    if custom_name:
        name = f"{custom_name}_{sha256_hex[:6]}{suffix}"
    else:
        name = f"{query_code}_{sha256_hex[:8]}{suffix}"
    return output_dir / name


def jitter_sleep(base: float, jitter: float = 0.5) -> None:
    time.sleep(base + random.random() * jitter)


def format_date_for_filename(date_str: str) -> str:
    """将日期字符串格式化为文件名格式，如 'Jan 26 2018' -> 'Jan262018'"""
    if not date_str:
        return "UnknownDate"
    # 移除空格并替换为下划线
    formatted = date_str.replace(" ", "")
    # 移除特殊字符，只保留字母数字
    formatted = re.sub(r'[^\w]', '', formatted)
    return formatted


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除或替换不安全的字符"""
    # 移除或替换文件名中的不安全字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 限制长度
    if len(filename) > 200:
        filename = filename[:200]
    return filename.strip()


def extract_party_names(text: str) -> str:
    """从文档描述中提取当事人名称，用于生成简短文件名。
    
    示例:
        "Brief of petitioners Masterpiece Cakeshop, et al. filed" -> "Masterpiece Cakeshop"
        "Reply of respondent Securities and Exchange Commission filed" -> "SEC"
        "Brief of amicus curiae American Civil Liberties Union" -> "ACLU"
    """
    if not text:
        return "Unknown"
    
    text = text.strip()
    
    # 常见缩写映射
    abbreviations = {
        'Securities and Exchange Commission': 'SEC',
        'Federal Trade Commission': 'FTC',
        'American Civil Liberties Union': 'ACLU',
        'National Association for the Advancement of Colored People': 'NAACP',
        'United States': 'US',
        'et al': '',
        'et al.': '',
    }
    
    # 提取模式：寻找 "of [petitioner/respondent/amicus] PARTY_NAME"
    patterns = [
        # "Brief/Reply of/for petitioners/respondents PARTY" 
        r'(?:reply\s+)?(?:brief|reply)\s+(?:of|for)\s+(?:petitioners?|respondents?)\s+(.+?)(?:\s+filed|\s*$)',
        # "Brief amici/amicus curiae of PARTY filed" — include optional 'of'
        r'(?:brief|reply)\s+(?:of|for)\s+amici\s+curiae\s+(?:of\s+)?(.+?)(?:\s+filed|\s+in\s+support|\s*$)',
        r'(?:brief|reply)\s+(?:of|for)\s+amicus\s+curiae\s+(?:of\s+)?(.+?)(?:\s+filed|\s+in\s+support|\s*$)',
        r'(?:brief|reply)\s+(?:of|for)\s+(.+?)\s+as\s+amicus\s+curiae',
    ]
    
    party_name = None
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            party_name = match.group(1).strip()
            # Strip trailing noise like "et al", "in support of", punctuation
            party_name = re.sub(r',?\s*et\s+al\.?', '', party_name, flags=re.IGNORECASE)
            party_name = re.sub(r'\s+in\s+support\s+of.+$', '', party_name, flags=re.IGNORECASE)
            party_name = party_name.strip(' .,')
            break
    
    if not party_name:
        # Fallback: skip generic noise words and take first meaningful words
        skip = {'brief', 'reply', 'amici', 'amicus', 'curiae', 'filed', 'main', 'document',
                'certificate', 'word', 'count', 'proof', 'service', 'of', 'for', 'the', 'and'}
        words_raw = text.split()
        words = [w.strip('.,;:') for w in words_raw if w.strip('.,;:').lower() not in skip and len(w.strip('.,;:')) > 1]
        party_name = ' '.join(words[:5]) if words else text.split()[0] if text.split() else 'Unknown'
    
    # 应用缩写
    for full_name, abbr in abbreviations.items():
        if full_name.lower() in party_name.lower():
            party_name = re.sub(full_name, abbr, party_name, flags=re.IGNORECASE)
    
    # 清理 "et al", "et al.", 逗号等
    party_name = re.sub(r',?\s*et\s+al\.?', '', party_name, flags=re.IGNORECASE)
    party_name = re.sub(r'\s*,\s*', ' ', party_name)  # 移除逗号
    party_name = re.sub(r'\s+', ' ', party_name).strip()
    
    # 限制长度，取前6个有意义的词（enough to distinguish parties)
    words = [w for w in party_name.split() if len(w) > 2 or w.isupper()]
    if len(words) > 6:
        party_name = ' '.join(words[:6])
    else:
        party_name = ' '.join(words) if words else party_name
    
    # 清理并限制总长度
    party_name = sanitize_filename(party_name)
    if len(party_name) > 70:
        party_name = party_name[:70].strip()
    
    return party_name if party_name else "Unknown"


# ----------------------------- 下载器 -----------------------------

class Downloader:
    def __init__(self, session: requests.Session, user_agent: Optional[str] = None, min_interval: float = 1.0):
        self.session = session
        self.min_interval = min_interval
        if user_agent:
            self.session.headers.update({'User-Agent': user_agent})

    def get(self, url: str, *, referer: Optional[str] = None, stream: bool = False, max_attempts: int = 3) -> requests.Response:
        headers: Dict[str, str] = {}
        if referer:
            headers['Referer'] = referer
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
                sleep = min(30, (2 ** (attempt - 1)))
                jitter_sleep(sleep, jitter=0.5)
        raise last_exc or requests.RequestException('request failed')

    def download_pdf(self, url: str, *, referer: Optional[str] = None, max_size_mb: int = 100) -> Tuple[bytes, Dict[str, Any]]:
        resp = self.get(url, referer=referer, stream=True)
        ctype = (resp.headers.get('Content-Type') or '')
        if 'pdf' not in ctype.lower():
            logging.warning('Unexpected Content-Type: %s', ctype)
        total = 0
        chunks: List[bytes] = []
        limit = max_size_mb * 1024 * 1024
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total > limit:
                    resp.close()
                    raise UnexpectedContentType(f'File too large: > {max_size_mb} MB')
        data = b''.join(chunks)
        meta = {
            'status_code': resp.status_code,
            'headers': dict(resp.headers),
            'content_type': ctype,
            'size_bytes': total,
        }
        jitter_sleep(self.min_interval)
        return data, meta

# ----------------------------- PDF 解析 -----------------------------

class PDFExtractor:
    def __init__(self, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr

    def extract_text_pdf(self, pdf_bytes: bytes) -> Tuple[List[str], str, str]:
        # 优先 pdfminer，回退 PyMuPDF
        if pdfminer_extract_text is not None:
            try:
                text = pdfminer_extract_text(io.BytesIO(pdf_bytes)) or ''
                pages = [p.strip() for p in re.split(r"\f|\n\s*\n\s*\n", text) if p.strip()]
                return pages, text, 'pdfminer'
            except Exception as e:
                logging.exception('pdfminer extract failed: %s', e)
        if fitz is not None:
            try:
                with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:  # type: ignore
                    pg_texts = [page.get_text() for page in doc]
                pages = [p.strip() for p in pg_texts if p and p.strip()]
                return pages, "\n\n".join(pg_texts), 'pymupdf'
            except Exception as e:
                logging.exception('pymupdf extract failed: %s', e)
        return [], '', 'none'

    def ocr_with_ocrmypdf(self, pdf_bytes: bytes) -> Tuple[List[str], str, str]:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'src.pdf'
            dst = Path(td) / 'ocr.pdf'
            src.write_bytes(pdf_bytes)
            try:
                subprocess.run([
                    'ocrmypdf', '--sidecar', str(Path(td)/'out.txt'), '--force-ocr', str(src), str(dst)
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                full = (Path(td)/'out.txt').read_text(encoding='utf-8', errors='ignore')
                pages = [p.strip() for p in re.split(r"\f|\n\s*\n\s*\n", full) if p.strip()]
                return pages, full, 'ocr: ocrmypdf/tesseract'
            except Exception as e:
                logging.exception('OCR failed: %s', e)
                return [], '', 'ocr: failed'

    def extract(self, pdf_bytes: bytes) -> Dict[str, Any]:
        pages, full, engine = self.extract_text_pdf(pdf_bytes)
        method = 'text'
        notes = engine
        if len(full.strip()) == 0 and self.enable_ocr:
            pages, full, notes = self.ocr_with_ocrmypdf(pdf_bytes)
            method = 'ocr' if len(full.strip()) > 0 else 'none'
        return {"pages": pages, "full": full, "method": method, "notes": notes}

# ----------------------------- 站点适配器（SCOTUS） -----------------------------

def _normalize_scotus_code_for_path(raw: str) -> Tuple[str, List[str]]:
    """将输入案号规范为 SCOTUS docket 文件名（不含 .htm），并给出备选方案。
    仅把“末尾 的 A+数字”替换为小写 a（例：24A949 → 24a949）；其余大小写保持原样。
    返回 (primary, alts)
    """
    s = (raw or "").strip()
    s = s.replace("–", "-").replace("—", "-")
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

class SiteAdapter:
    def __init__(self, session: requests.Session, base_url: str):
        self.session = session
        self.base_url = base_url.rstrip('/')

    def search_and_resolve(self, query_code: str) -> Optional[ResolvedDoc]:
        raise NotImplementedError

class ScotusDocketAdapter(SiteAdapter):
    """只处理 YY-数字（如 75-1552）；忽略 24A***。"""
    def _build_url_from_code(self, code_no_ext: str) -> str:
        # 目标格式： https://www.supremecourt.gov/search.aspx?filename=/docketfiles/06-5754.htm
        return f"{self.base_url}/search.aspx?filename=/docketfiles/{code_no_ext}.htm"

    def _build_alt_url_from_code(self, code_no_ext: str) -> str:
        # Direct docket HTML: used for post-2019 cases (reliable; avoids search.aspx wrapper)
        # e.g. https://www.supremecourt.gov/docket/docketfiles/html/public/20-843.html
        return f"{self.base_url}/docket/docketfiles/html/public/{code_no_ext}.html"

    def _is_blank_search_results_page(self, soup: BeautifulSoup) -> bool:
        """启发式：识别空的“Search Results”页（常见于错误格式）。
        条件：包含“Search Results/Search”但不含“Docket for”，且无任何 PDF 链接。
        """
        try:
            text = soup.get_text(" ", strip=True)
        except Exception:
            text = ''
        has_search_ui = ('Search Results' in text) or ('Search' in text)
        has_no = 'No.' not in text
        return has_search_ui and (not has_no)

    def _iter_pdf_links(self, soup: BeautifulSoup):
        """Iterate over Main Document PDF links, yielding (href, desc, parent_text, date_text, row_text).
        parent_text = the <td> cell text containing the link cluster.
        row_text    = the full <tr> row text, which starts with the document-type label.
        """
        for a in soup.find_all('a'):
            href = (a.get('href') or '').strip()
            if not href:
                continue
            low = href.lower()
            if not (low.endswith('.pdf') or ('/docketpdf/' in low)):
                continue
            desc = a.get_text(" ", strip=True)
            logging.debug('Found PDF link: desc="%s", href="%s"', desc, href)
            if desc.strip().lower() != 'main document':
                logging.debug('Skipping non-Main Document: "%s"', desc)
                continue
            logging.info('Processing Main Document: "%s"', desc)
            parent_text = desc
            date_text = ""
            row_text = ""
            current = a.find_parent()
            while current:
                if current.name == 'td':
                    parent_text = current.get_text(" ", strip=True)
                    if current.find_previous_sibling('td', class_='ProceedingDate'):
                        date_cell = current.find_previous_sibling('td', class_='ProceedingDate')
                        date_text = date_cell.get_text(" ", strip=True) if date_cell else ""
                    # Get the full row text for document-type identification
                    row = current.find_parent('tr')
                    if row:
                        row_text = row.get_text(" ", strip=True)
                    break
                current = current.find_parent()
            yield href, desc, parent_text, date_text, row_text

    def search_and_resolve(self, query_code: str) -> Optional[ResolvedDoc]:
        if not re.match(r"^[0-9]{2}-[0-9]+$", (query_code or "").strip()):
            return None
        primary, alts = _normalize_scotus_code_for_path(query_code)
        candidates = [primary] + alts
        last_resp = None
        for cand in candidates:
            # Try new HTML format first (post-2019 cases), then old HTM format (pre-2020)
            for url in (self._build_alt_url_from_code(cand), self._build_url_from_code(cand)):
                r = self.session.get(url, timeout=30, allow_redirects=True)
                last_resp = r
                if r.status_code != 200:
                    continue
                final_url = r.url or url
                # Accept .htm or .html docket pages (both old and new SCOTUS formats).
                # SCOTUS serves via search.aspx?filename=...html — endswith still works
                # because the query param ends with the docket filename.
                low = final_url.lower()
                if not (low.endswith(f"/{cand.lower()}.htm") or low.endswith(f"/{cand.lower()}.html")
                        or low.endswith(f"{cand.lower()}.htm") or low.endswith(f"{cand.lower()}.html")):
                    continue
                soup = BeautifulSoup(r.text, 'lxml')
                if self._is_blank_search_results_page(soup):
                    # 错误格式引导到空的搜索结果页，尝试下一个 URL 变体
                    continue
                anchors = list(self._iter_pdf_links(soup))
                if not anchors:
                    return ResolvedDoc(query_code, final_url, '', (soup.find('title').get_text(strip=True) if soup.find('title') else None), None, {"note": "no pdf link on page"})
                # Fix: replaced 'search_and_resolve' unpack from 3 to 4 values + ignored row_text
                href, desc, parent_text, date_text, row_text = anchors[0]
                download_url = href if href.startswith('http') else f"{self.base_url}{href if href.startswith('/') else '/' + href}"
                title = (desc or parent_text or '').strip() or (soup.find('title').get_text(strip=True) if soup.find('title') else None)
                return ResolvedDoc(query_code, final_url, download_url, title, None, {})
        if last_resp is not None:
            logging.info("SCOTUS page not resolved for %s; last url=%s status=%s", query_code, last_resp.url, last_resp.status_code)
        return None

    def collect_briefs(self, query_code: str) -> List[ResolvedDoc]:
        """解析 docket 页面，提取最终庭审阶段的 party briefs 与 amicus briefs。
        过滤掉涉及 certiorari 阶段和无关文档（如 petition, motions, appendix 等）。
        """
        primary, alts = _normalize_scotus_code_for_path(query_code)
        candidates = [primary] + alts
        docs: List[ResolvedDoc] = []
        last_resp = None
        found_valid_page = False
        for cand in candidates:
            if found_valid_page:
                break
            # Try new HTML format first (post-2019 cases), then old HTM format (pre-2020)
            for url in (self._build_alt_url_from_code(cand), self._build_url_from_code(cand)):
                r = self.session.get(url, timeout=30, allow_redirects=True)
                last_resp = r
                if r.status_code != 200:
                    continue
                final_url = r.url or url
                # Accept .htm or .html docket pages (old and new SCOTUS URL formats).
                low = final_url.lower()
                if not (low.endswith(f"/{cand.lower()}.htm") or low.endswith(f"/{cand.lower()}.html")
                        or low.endswith(f"{cand.lower()}.htm") or low.endswith(f"{cand.lower()}.html")):
                    continue
                soup = BeautifulSoup(r.text, 'lxml')
                if self._is_blank_search_results_page(soup):
                    continue

                # 找到有效页面 —— 在此处处理所有 PDF 链接（修复：之前错误地放在循环外）
                found_valid_page = True

                # Heuristic: merits-stage briefs usually appear as entries containing "Brief" with
                # party or amicus indicators, and without cert-stage words.
                include_kw = [
                    'brief of petitioner', 'brief for petitioner', 'brief of petitioners', 'brief for petitioners',
                    'brief of respondent', 'brief for respondent', 'brief of respondents', 'brief for respondents',
                    'amicus', 'amici', 'amicus curiae', 'amici curiae', 'reply brief'
                ]

                def is_merits_brief(text: str) -> bool:
                    t = (text or '').strip().lower()
                    if 'brief' not in t:
                        return False
                    if any(x in t for x in include_kw):
                        return True
                    return False

                # Strict allowlist: only download documents whose row label starts with a brief/reply type.
                # Use row_text (full <tr>) so we get the actual document-type label, not just the
                # surrounding cell text that might mention 'brief' incidentally (e.g. motions).
                BRIEF_PREFIXES = (
                    'brief of petitioner', 'brief for petitioner',
                    'brief of respondent', 'brief for respondent',
                    'brief of amicus', 'brief for amicus',
                    'brief of amici', 'brief for amici',
                    'reply brief', 'reply of petitioner', 'reply for petitioner',
                    'reply of respondent', 'reply for respondent',
                    # Common shorthand labels on SCOTUS docket pages:
                    'brief amicus curiae', 'brief amici curiae',
                    # Catch-all for labelled briefs not matching above:
                )

                for href, desc, parent_text, date_text, row_text in self._iter_pdf_links(soup):
                    row_lower = row_text.lower()
                    # The row_text starts with the date, then the document type.
                    # Extract just the document-type portion (after the date-like prefix).
                    # We check if ANY brief prefix appears in the first 120 chars of row_text
                    # (to capture the label before the link cluster and word counts).
                    row_label = row_lower[:120]
                    is_brief = any(p in row_label for p in BRIEF_PREFIXES)
                    if not is_brief:
                        logging.info('Skipping non-brief document: "%s"', row_text[:80])
                        continue

                    full_url = href if href.startswith('http') else f"{self.base_url}{href if href.startswith('/') else '/' + href}"
                    title_text = desc or parent_text

                    # Build descriptive filename: {date}_{role}_{party}
                    # date_part: compact YYYYMMDD from date_text (e.g. "Jul 20 2021" → "20210720")
                    date_part = ""
                    if date_text:
                        try:
                            import datetime as _dt
                            date_part = _dt.datetime.strptime(date_text.strip(), "%b %d %Y").strftime("%Y%m%d")
                        except Exception:
                            date_part = re.sub(r'\s+', '', date_text)[:8]

                    # role: brief type extracted from the row label
                    _rl = row_text.lower()[:80]
                    if 'reply' in _rl:
                        role = 'reply'
                    elif 'petitioner' in _rl:
                        role = 'petitioner'
                    elif 'respondent' in _rl:
                        role = 'respondent'
                    elif 'amici' in _rl:
                        role = 'amici'
                    elif 'amicus' in _rl:
                        role = 'amicus'
                    else:
                        role = 'brief'

                    party_name = extract_party_names(parent_text)
                    parts = [p for p in [date_part, role, party_name] if p]
                    # Replace spaces with underscores for clean filenames
                    custom_filename = sanitize_filename('_'.join(parts).replace(' ', '_'))[:90]
                    logging.info('Filename: "%s"', custom_filename)
                    docs.append(ResolvedDoc(query_code, final_url, full_url, title_text, None, {
                        "date": date_text,
                        "custom_filename": custom_filename
                    }))

                break  # stop trying more URL variants once a valid page is found

        if not docs and last_resp is not None:
            logging.info('No merits briefs found for %s; last url=%s status=%s', query_code, last_resp.url, last_resp.status_code)
        else:
            logging.info('Found %d brief/reply documents for %s', len(docs), query_code)
        return docs

# ----------------------------- 主流程 -----------------------------

def load_queries_from_json(path: Path) -> List[Dict[str, Any]]:
    """读取 JSON（数组/对象）或 JSON Lines（.jsonl）。每条记录应至少包含 docket_no。"""
    text = path.read_text(encoding='utf-8')
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception:
        items: List[Dict[str, Any]] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
        return items
    return []


def discover_cases_from_oyez(session: requests.Session, term: Optional[int] = None,
                              page_size: int = 100, min_interval: float = 1.0):
    """Generator: page through the Oyez API and yield docket numbers (XX-NNNN format only).

    Args:
        session:      requests.Session to reuse.
        term:         If set, only yield cases from that SCOTUS term year (e.g. 2023).
        page_size:    Cases per page (max 100 recommended).
        min_interval: Seconds to sleep between API pages.
    Yields:
        str: docket number, e.g. '22-300'
    """
    base = 'https://api.oyez.org/cases'
    page = 0
    seen: set = set()
    docket_re = re.compile(r'^[0-9]{2}-[0-9]+$')

    while True:
        params: dict = {'per_page': page_size, 'page': page}
        if term is not None:
            params['filter'] = f'term:{term}'
        try:
            r = session.get(base, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logging.error('Oyez API error on page %d: %s', page, e)
            break

        if not data:
            break  # no more pages

        found_any = False
        for case in data:
            docket = str(case.get('docket_number') or '').strip()
            if not docket or not docket_re.match(docket):
                continue  # skip old-format dockets (plain numbers) and A-numbers
            if re.match(r'^[0-9]{2}A[0-9]+$', docket, re.IGNORECASE):
                continue
            if docket in seen:
                continue
            seen.add(docket)
            found_any = True
            yield docket

        logging.info('Oyez page %d: fetched %d cases, yielded %d new dockets so far',
                     page, len(data), len(seen))
        page += 1
        time.sleep(min_interval)

        if not found_any and len(data) < page_size:
            break  # last page, nothing new

@dataclass
class RunStats:
    total: int = 0
    success: int = 0
    not_found: int = 0
    failed: int = 0


def build_adapter(site_key: str, session: requests.Session, base_url: Optional[str]) -> SiteAdapter:
    site_key = site_key.lower()
    if site_key == 'scotus':
        if not base_url:
            base_url = 'https://www.supremecourt.gov'
        return ScotusDocketAdapter(session, base_url)
    raise ValueError(f"Unknown site key: {site_key}")


def init_logger(out_dir: Path) -> None:
    # Write logs to output/logs/ instead of data/logs/
    project_root = Path(__file__).resolve().parents[1]
    output_logs_dir = project_root / 'output' / 'logs'
    ensure_dir(output_logs_dir)
    log_path = output_logs_dir / 'scraper_output.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding='utf-8')
        ]
    )


def write_json(output_dir: Path, query_code: str, sha256_hex: str, payload: Dict[str, Any], custom_name: Optional[str] = None) -> Path:
    # 创建案件文件夹下的json子文件夹
    case_dir = output_dir / query_code / 'json'
    path = make_output_paths(case_dir, query_code, sha256_hex, '.json', custom_name)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def write_pdf(output_dir: Path, query_code: str, sha256_hex: str, data: bytes, custom_name: Optional[str] = None) -> Path:
    # 创建案件文件夹下的pdf子文件夹
    case_dir = output_dir / query_code / 'pdf'
    path = make_output_paths(case_dir, query_code, sha256_hex, '.pdf', custom_name)
    save_bytes(path, data)
    return path


def process_one(query_code: str, adapter: SiteAdapter, downloader: Downloader, extractor: PDFExtractor, output_dir: Path) -> Tuple[str, str]:
    try:
        # SCOTUS: 尝试收集并下载所有最终庭审阶段的 party/amicus briefs
        brief_docs: List[ResolvedDoc] = []
        if isinstance(adapter, ScotusDocketAdapter):
            try:
                brief_docs = adapter.collect_briefs(query_code)
            except Exception as e:
                logging.exception('collect_briefs failed for %s: %s', query_code, e)
        
        # 如果没有找到任何 brief，跳过此案件
        if not brief_docs:
            return 'not_found', 'no merits briefs found'
        
        logging.info('Processing %d brief/reply documents for case %s', len(brief_docs), query_code)
        results: List[str] = []
        targets = brief_docs

        for i, resolved in enumerate(targets, 1):
            pdf_bytes, meta = downloader.download_pdf(resolved.download_url, referer=resolved.source_page_url)
            sha = sha256_bytes(pdf_bytes)
            
            # 使用自定义文件名
            custom_name = None
            if resolved.extra and 'custom_filename' in resolved.extra:
                custom_name = resolved.extra['custom_filename']
            
            pdf_path = write_pdf(output_dir, query_code, sha, pdf_bytes, custom_name)
            pages_count = 0
            if fitz is not None:
                try:
                    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:  # type: ignore
                        pages_count = doc.page_count
                except Exception:
                    pages_count = 0
            extr = extractor.extract(pdf_bytes)
            payload = {
                'version': JSON_SCHEMA['version'],
                'document': {
                    'query_code': query_code,
                    'source_page_url': resolved.source_page_url,
                    'download_url': resolved.download_url,
                    'fetched_at': dt.datetime.utcnow().isoformat() + 'Z',
                    'sha256': sha,
                    'filename': pdf_path.name,
                    'file_size_bytes': len(pdf_bytes),
                    'pages': pages_count,
                    'extraction': {
                        'method': extr['method'],
                        'chars': len(extr['full'] or ''),
                        'notes': extr['notes'],
                    },
                    'metadata': {
                        'title': resolved.title,
                        'date': resolved.extra.get('date') if resolved.extra else None,
                        'extra': resolved.extra or {},
                    },
                    'content': {
                        'page_text': extr['pages'],
                        'full_text': extr['full'],
                    }
                }
            }
            json_path = write_json(output_dir, query_code, sha, payload, custom_name)
            logging.info('SUCCESS %s [%d/%d] → %s', query_code, i, len(targets), json_path)
            results.append(str(json_path))
        return ('success', ';'.join(results)) if results else ('not_found', 'no briefs found')
    except NotFoundError as e:
        logging.warning('NOT_FOUND %s: %s', query_code, e)
        return 'not_found', str(e)
    except SiteBlockedError as e:
        logging.error('BLOCKED %s: %s', query_code, e)
        raise
    except Exception as e:
        logging.exception('FAILED %s: %s', query_code, e)
        return 'failed', str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description='SCOTUS brief crawler')
    ap.add_argument('--queries-json', type=Path, default=None,
                    help='JSON/JSONL input file with docket_no fields (omit to use --all-cases)')
    ap.add_argument('--all-cases', action='store_true',
                    help='Auto-discover all cases via the Oyez API')
    ap.add_argument('--term', type=int, nargs='+', default=None, metavar='YEAR',
                    help='With --all-cases: one year (--term 2023) or a range (--term 1997 2003)')
    ap.add_argument('--output-dir', type=Path, default=Path('./data'),
                    help='Directory to write PDFs and JSON (default: ./output)')
    ap.add_argument('--min-interval', type=float, default=1.5,
                    help='Seconds between requests (default: 1.5)')
    ap.add_argument('--min-year', type=int, default=1900,
                    help='Skip cases before this year, e.g. --min-year 2000 (default: 1900 = all)')
    args = ap.parse_args()

    if not args.all_cases and args.queries_json is None:
        ap.error('Provide either --queries-json <file> or --all-cases')

    ensure_dir(args.output_dir)
    init_logger(args.output_dir)

    user_agent = 'crawler-pdf-json/1.0'
    session = requests.Session()
    session.headers.update({
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    })

    adapter = build_adapter('scotus', session, 'https://www.supremecourt.gov')
    downloader = Downloader(session, user_agent=user_agent, min_interval=args.min_interval)
    extractor = PDFExtractor(enable_ocr=False)


    # ---- build task iterator ------------------------------------------------
    def _filter_docket(code: str) -> bool:
        """Return True if docket passes format and year filters."""
        if re.match(r'^[0-9]{2}A[0-9]+$', code, re.IGNORECASE):
            return False  # skip A-numbers
        if not re.match(r'^[0-9]{2}-[0-9]+$', code):
            return False
        if args.min_year > 0:
            try:
                yy = int(code.split('-', 1)[0])
                current_yy = int(dt.datetime.utcnow().strftime('%y'))
                full_year = 2000 + yy if yy <= current_yy else 1900 + yy
                if full_year < args.min_year:
                    return False
            except Exception:
                return False
        return True

    if args.all_cases:
        # Resolve term range
        term_years: List[int] = []
        if args.term:
            if len(args.term) == 1:
                term_years = [args.term[0]]
                logging.info('Discovering cases for SCOTUS term %d via Oyez API…', args.term[0])
            elif len(args.term) == 2:
                start, end = sorted(args.term)
                term_years = list(range(start, end + 1))
                logging.info('Discovering cases for SCOTUS terms %d–%d via Oyez API…', start, end)
            else:
                ap.error('--term accepts 1 or 2 year values')
        else:
            logging.info('Discovering ALL cases via Oyez API (this may take a while)…')

        def _all_cases_gen():
            if term_years:
                seen_global: set = set()
                for yr in term_years:
                    for code in discover_cases_from_oyez(session, term=yr, min_interval=args.min_interval):
                        if code not in seen_global and _filter_docket(code):
                            seen_global.add(code)
                            yield code
            else:
                for code in discover_cases_from_oyez(session, term=None, min_interval=args.min_interval):
                    if _filter_docket(code):
                        yield code

        tasks_iter = _all_cases_gen()
    else:
        records = load_queries_from_json(args.queries_json)
        raw: List[str] = []
        for rec in records:
            code = str(rec.get('docket_no', '')).strip()
            if code and _filter_docket(code):
                raw.append(code)
            elif code:
                logging.info('SKIP unsupported/filtered docket: %s', code)
        tasks_iter = iter(raw)

    # ---- run ----------------------------------------------------------------
    stats = RunStats()
    for q in tasks_iter:
        stats.total += 1
        status, msg = process_one(q, adapter, downloader, extractor, args.output_dir)
        if status == 'success':
            stats.success += 1
        elif status == 'not_found':
            logging.info('SKIP %s: %s', q, msg)
            stats.not_found += 1
        else:
            stats.failed += 1

    logging.info('RUN DONE: total=%d success=%d not_found=%d failed=%d',
                 stats.total, stats.success, stats.not_found, stats.failed)


if __name__ == '__main__':
    main()
