#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#################################################################### 
# 使用方法：使用terminal运行以下命令
# 例如："分析案例17-773和17-21:
# python3 process_similarity.py --cases 17-773 17-21 --output "test_results.jsonl" 
####################################################################

# Issues:
# 1. 目前是手动排除了amicus brief，后续可能改成在clean阶段直接排除
# 2. chunk size和overlap不确定，暂定数值
# 3. 暂定使用TfidfVectorizer向量化，计算余弦相似度
# 4. 参数暂定，后续可能调整
# 5. 相似度过低，可能由于包含了非party brief，或者包含了非指向相同内容的party brief，后续可能需要检查文本内容



def read_json(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def extract_oral_argument_text(corpus_json_path: Path) -> str:
    data = read_json(corpus_json_path)
    # In corpus files, transcript is under key 'utterence' (note spelling in dataset)
    text = data.get('utterence') or ''
    return text


def extract_brief_text(brief_json_path: Path) -> str:
    data = read_json(brief_json_path)
    # Files may nest under 'document' -> 'content'
    content = (data.get('content') or {})
    if not content:
        document = data.get('document') or {}
        content = (document.get('content') or {})
    # Some files may use 'extraction' -> 'content'
    if not content:
        content = (data.get('extraction') or {}).get('content') or {}
    full_text = content.get('full_text')
    if isinstance(full_text, str) and full_text.strip():
        return full_text
    page_text = content.get('page_text')
    if isinstance(page_text, list):
        try:
            return '\n'.join([p for p in page_text if isinstance(p, str)])
        except Exception:
            pass
    # Fallback: look for common keys at root
    for key in ['full_text', 'text', 'body', 'extracted_text', 'ocr_text']:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ''

#手动排除了amicus brief，后续可能改成在clean阶段直接排除
def is_party_brief(filename: str) -> bool:
    name = filename.lower()
    # Exclude non-party items
    if 'amicus' in name or 'motion' in name or 'letter' in name:
        return False
    # Include any briefs or replies by parties
    if 'brief' in name or 'reply' in name:
        return True
    return False

#chunk size和overlap不确定，暂定数值
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    # Normalize whitespace
    clean = re.sub(r'[ \t\u00A0]+', ' ', text)
    clean = re.sub(r'\s*\n\s*', '\n', clean)
    tokens = clean.split()
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        if not chunk_tokens:
            break
        chunks.append(' '.join(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break
        i += max(1, chunk_size - overlap)
    return chunks

#暂定使用TfidfVectorizer向量化，计算余弦相似度
def compute_avg_cross_similarity(brief_chunks: List[str], oral_chunks: List[str]) -> float:
    if not brief_chunks or not oral_chunks:
        return float('nan')
    docs = brief_chunks + oral_chunks
    #参数暂定，后续可能调整
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    X = vectorizer.fit_transform(docs)
    n_b = len(brief_chunks)
    n_o = len(oral_chunks)
    B = X[:n_b]
    O = X[n_b:]
    sims = cosine_similarity(B, O)
    
    # 检查是否有NaN值，如果有则抛出异常
    if np.isnan(sims).any():
        raise ValueError("Found NaN values in similarity matrix - this indicates data quality issues")
    
    return float(np.mean(sims))


def process_case(case_dir: Path, chunk_size: int, overlap: int) -> Dict:
    json_dir = case_dir / 'json'
    if not json_dir.exists():
        return {'case': case_dir.name, 'error': 'missing json dir'}
    # Find corpus
    corpus_files = list(json_dir.glob(f"{case_dir.name}__corpus.json"))
    if not corpus_files:
        return {'case': case_dir.name, 'error': 'missing corpus json'}
    
    try:
        oral_text = extract_oral_argument_text(corpus_files[0])
        oral_chunks = chunk_text(oral_text, chunk_size=chunk_size, overlap=overlap)
    except Exception as e:
        return {'case': case_dir.name, 'error': f'oral extraction failed: {str(e)}'}

    # Collect party briefs
    brief_texts: List[Tuple[str, str]] = []  # (filename, text)
    candidate_files = 0
    empty_text_files: List[str] = []
    for p in sorted(json_dir.glob('*.json')):
        if p.name.endswith('__corpus.json'):
            continue
        if not is_party_brief(p.name):
            continue
        candidate_files += 1
        t = extract_brief_text(p)
        if t and t.strip():
            brief_texts.append((p.name, t))
        else:
            empty_text_files.append(p.name)

    brief_chunks = []
    brief_chunk_map = []  # (brief_filename, chunk_index)
    for fname, text in brief_texts:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, c in enumerate(chunks):
            brief_chunks.append(c)
            brief_chunk_map.append((fname, idx))

    try:
        avg_sim = compute_avg_cross_similarity(brief_chunks, oral_chunks)
    except Exception as e:
        return {
            'case': case_dir.name,
            'error': f'similarity computation failed: {str(e)}',
            'num_brief_docs': len(brief_texts),
            'num_candidate_brief_files': candidate_files,
            'num_empty_brief_files': len(empty_text_files),
            'num_brief_chunks': len(brief_chunks),
            'num_oral_chunks': len(oral_chunks),
        }

    return {
        'case': case_dir.name,
        'num_brief_docs': len(brief_texts),
        'num_candidate_brief_files': candidate_files,
        'num_empty_brief_files': len(empty_text_files),
        'num_brief_chunks': len(brief_chunks),
        'num_oral_chunks': len(oral_chunks),
        'avg_brief_oral_cosine': avg_sim,
        'empty_brief_samples': empty_text_files[:3],
    }


def main():
    parser = argparse.ArgumentParser(description='Compute average brief-oral similarity for cases')
    parser.add_argument('--data-root', type=str, required=False,
                        default=str(Path(__file__).resolve().parents[1] / 'data'),
                        help='Root directory containing case folders (e.g., data)')
    parser.add_argument('--cases', type=str, nargs='*', default=['17-773', '17-21'],
                        help='Case docket numbers to process')
    parser.add_argument('--chunk-size', type=int, default=1200)
    parser.add_argument('--overlap', type=int, default=200)
    parser.add_argument('--output', type=str, default='similarity_results.jsonl')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results = []
    for case in args.cases:
        case_dir = data_root / case
        res = process_case(case_dir, chunk_size=args.chunk_size, overlap=args.overlap)
        results.append(res)

    out_path = Path(args.output).resolve()
    with out_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Also print a concise summary
    for r in results:
        print(f"{r['case']}: docs={r.get('num_brief_docs')} brief_chunks={r.get('num_brief_chunks')} oral_chunks={r.get('num_oral_chunks')} avg={r.get('avg_brief_oral_cosine')}")


if __name__ == '__main__':
    main()


