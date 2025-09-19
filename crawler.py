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
    python crawler_pdf_to_json.py \
      --queries-json cases.jsonl \
      --output-dir ./data \
      --site scotus

依赖（最小集）
    pip install requests beautifulsoup4 lxml pymupdf
    # 可选：更高质量文本抽取
    pip install pdfminer.six
    # 可选：OCR（本机需装 ocrmypdf/tesseract）

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
    """创建输出路径"""
    if custom_name:
        name = f"{custom_name}{suffix}"
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
        # 另一已知格式： https://www.supremecourt.gov/search.aspx?filename=/docket/docketfiles/html/public/17-1201.html
        return f"{self.base_url}/search.aspx?filename=/docket/docketfiles/html/public/{code_no_ext}.html"

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
        """以更鲁棒的方式遍历 PDF 链接（大小写不敏感，兼容 DocketPDF 目录）。只返回 Main Document 链接。"""
        for a in soup.find_all('a'):
            href = (a.get('href') or '').strip()
            if not href:
                continue
            low = href.lower()
            if low.endswith('.pdf') or ('/docketpdf/' in low):
                desc = a.get_text(" ", strip=True)
                # 调试日志：显示所有找到的PDF链接
                logging.debug('Found PDF link: desc="%s", href="%s"', desc, href)
                # 只处理 "Main Document" 链接
                if desc.strip().lower() != 'main document':
                    logging.debug('Skipping non-Main Document: "%s"', desc)
                    continue
                logging.info('Processing Main Document: "%s"', desc)
                # 获取更广泛的父元素文本，包括整个表格单元格
                parent_text = desc
                date_text = ""
                current = a.find_parent()
                while current:
                    if current.name == 'td':  # 找到表格单元格
                        parent_text = current.get_text(" ", strip=True)
                        # 查找同一行中的日期单元格
                        if current.find_previous_sibling('td', class_='ProceedingDate'):
                            date_cell = current.find_previous_sibling('td', class_='ProceedingDate')
                            date_text = date_cell.get_text(" ", strip=True) if date_cell else ""
                        break
                    current = current.find_parent()
                yield href, desc, parent_text, date_text

    def search_and_resolve(self, query_code: str) -> Optional[ResolvedDoc]:
        if not re.match(r"^[0-9]{2}-[0-9]+$", (query_code or "").strip()):
            return None
        primary, alts = _normalize_scotus_code_for_path(query_code)
        candidates = [primary] + alts
        last_resp = None
        for cand in candidates:
            for url in (self._build_url_from_code(cand), self._build_alt_url_from_code(cand)):
                r = self.session.get(url, timeout=30, allow_redirects=True)
                last_resp = r
                if r.status_code != 200:
                    continue
                final_url = r.url or url
                # 接受两种末尾形式：.htm 或 .html
                low = final_url.lower()
                if not (low.endswith(f"/{cand.lower()}.htm") or low.endswith(f"/{cand.lower()}.html")):
                    continue
                soup = BeautifulSoup(r.text, 'lxml')
                if self._is_blank_search_results_page(soup):
                    # 错误格式引导到空的搜索结果页，尝试下一个 URL 变体
                    continue
                anchors = list(self._iter_pdf_links(soup))
                if not anchors:
                    return ResolvedDoc(query_code, final_url, '', (soup.find('title').get_text(strip=True) if soup.find('title') else None), None, {"note": "no pdf link on page"})
                href, desc, parent_text = anchors[0]
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
        for cand in candidates:
            for url in (self._build_url_from_code(cand), self._build_alt_url_from_code(cand)):
                r = self.session.get(url, timeout=30, allow_redirects=True)
                last_resp = r
                if r.status_code != 200:
                    continue
                final_url = r.url or url
                low = final_url.lower()
                if not (low.endswith(f"/{cand.lower()}.htm") or low.endswith(f"/{cand.lower()}.html")):
                    continue
                soup = BeautifulSoup(r.text, 'lxml')
                if self._is_blank_search_results_page(soup):
                    continue

            # Heuristic: merits-stage briefs usually appear as entries containing "Brief" with
            # party or amicus indicators, and without cert-stage words.
            include_kw = [
                'brief of petitioner', 'brief for petitioner', 'brief of petitioners', 'brief for petitioners',
                'brief of respondent', 'brief for respondent', 'brief of respondents', 'brief for respondents',
                'amicus', 'amici', 'amicus curiae', 'amici curiae', 'reply brief'
            ]
            exclude_kw = []

            def is_merits_brief(text: str) -> bool:
                t = (text or '').strip().lower()
                if 'brief' not in t:
                    return False
                if any(x in t for x in include_kw):
                    return True
                return False

            # Prefer container rows that hold both date and description.
            # On SCOTUS pages, PDFs are linked inside <a> tags. We'll inspect the parent text.
            for href, desc, parent_text, date_text in self._iter_pdf_links(soup):
                # _iter_pdf_links 已经确保只返回 "Main Document" 链接，现在检查是否包含 brief 或 reply
                parent_text_lower = parent_text.lower()
                if not ('brief' in parent_text_lower or 'reply' in parent_text_lower):
                    logging.info('Skipping document (no brief/reply): "%s"', parent_text[:100] + '...' if len(parent_text) > 100 else parent_text)
                    continue
                
                logging.info('Processing brief/reply document: "%s"', parent_text[:100] + '...' if len(parent_text) > 100 else parent_text)
                
                full_url = href if href.startswith('http') else f"{self.base_url}{href if href.startswith('/') else '/' + href}"
                title_text = desc or parent_text
                # 创建自定义文件名：date_name
                formatted_date = format_date_for_filename(date_text)
                logging.debug('Original parent_text: "%s"', parent_text)
                
                # 更精确地提取文档描述：找到 <br> 标签之前的内容
                clean_title = parent_text
                # 如果包含 <br> 标签，只取之前的部分
                if '<br' in clean_title:
                    clean_title = clean_title.split('<br')[0]
                
                # 移除所有链接相关的文本
                link_patterns = [
                    r'Main Document',
                    r'Certificate of Word Count', 
                    r'Proof of Service',
                    r'Other',
                    r'\(Distributed\)',
                    r'<span[^>]*>.*?</span>',  # 移除span标签及其内容
                    r'<a[^>]*>.*?</a>',  # 移除a标签及其内容
                ]
                
                for pattern in link_patterns:
                    clean_title = re.sub(pattern, '', clean_title, flags=re.IGNORECASE)
                
                # 清理多余的空格、换行和特殊字符
                clean_title = re.sub(r'\s+', ' ', clean_title).strip()
                clean_title = clean_title.strip('.,;:')
                clean_title = sanitize_filename(clean_title)
                
                custom_filename = f"{formatted_date}_{clean_title}"
                logging.info('Generated filename: "%s"', custom_filename)
                docs.append(ResolvedDoc(query_code, final_url, full_url, title_text, None, {
                    "date": date_text,
                    "custom_filename": custom_filename
                }))
                # 移除 break 语句，继续处理所有符合条件的文档
            # 移除 break 语句，继续尝试其他 URL 变体
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
    ensure_dir(out_dir / 'logs')
    log_path = out_dir / 'logs' / 'app.log'
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries-json', type=Path, required=True, help='JSON/JSONL 输入文件，记录需含 docket_no')
    ap.add_argument('--output-dir', type=Path, default=Path('./data'))
    ap.add_argument('--site', type=str, required=True, help='站点键，例如 scotus')
    ap.add_argument('--base-url', type=str, default=None)
    ap.add_argument('--user-agent', type=str, default='crawler-pdf-json/1.0')
    ap.add_argument('--min-interval', type=float, default=1.0)
    ap.add_argument('--enable-ocr', type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    init_logger(args.output_dir)

    session = requests.Session()
    session.headers.update({'User-Agent': args.user_agent, 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'})

    adapter = build_adapter(args.site, session, args.base_url)
    downloader = Downloader(session, user_agent=args.user_agent, min_interval=args.min_interval)
    extractor = PDFExtractor(enable_ocr=bool(args.enable_ocr))

    # 从 JSON/JSONL 构建任务，仅保留 YY-数字案号
    records = load_queries_from_json(args.queries_json)
    tasks: List[str] = []
    for rec in records:
        code = str(rec.get('docket_no', '')).strip()
        if not code:
            continue
        if re.match(r'^[0-9]{2}A[0-9]+$', code, re.IGNORECASE):  # 跳过 A 号
            logging.info('SKIP application docket (A-number): %s', code)
            continue
        if not re.match(r'^[0-9]{2}-[0-9]+$', code):
            logging.info('SKIP unsupported docket format: %s', code)
            continue
        # 仅保留 2000 年及之后的案件（根据两位年份映射到完整年份）
        try:
            yy = int(code.split('-', 1)[0])
            current_yy = int(dt.datetime.utcnow().strftime('%y'))
            # 两位年份到完整年份：小于等于当前两位年视为 2000+yy，否则视为 1900+yy
            full_year = 2000 + yy if yy <= current_yy else 1900 + yy
            if full_year < 2001:
                logging.info('SKIP pre-2000 docket: %s (year=%d)', code, full_year)
                continue
        except Exception:
            logging.info('SKIP due to year parse error: %s', code)
            continue
        tasks.append(code)

    stats = RunStats(total=len(tasks))
    for q in tasks:
        status, msg = process_one(q, adapter, downloader, extractor, args.output_dir)
        if status == 'success':
            stats.success += 1
        elif status == 'not_found':
            if 'no merits briefs found' in msg:
                logging.info('SKIP %s: %s', q, msg)
            else:
                logging.info('NOT_FOUND %s: %s', q, msg)
            stats.not_found += 1
        else:
            stats.failed += 1

    logging.info('RUN DONE: total=%d success=%d not_found=%d failed=%d', stats.total, stats.success, stats.not_found, stats.failed)


if __name__ == '__main__':
    main()
