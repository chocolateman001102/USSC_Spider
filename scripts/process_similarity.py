#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#################################################################### 
# 使用方法：使用terminal运行以下命令
# 例如："分析案例17-773和17-21:
# python3 process_similarity.py --cases 17-773 17-21 --output "test_results.jsonl" 
####################################################################

# Issues:
# 1. 目前是手动排除了amicus brief，后续可能改成在clean阶段直接排除
# 2. chunk size和overlap不确定，暂定数值
# 3. 使用sentence-transformers神经网络嵌入向量，计算余弦相似度（替代了TF-IDF）
# 4. 使用最大相似度平均法：对于每个brief chunk，找到与所有oral chunks的最大相似度，然后平均
# 5. 相似度过低，可能由于包含了 非party brief，或者包含了非指向相同内容的party brief，后续可能需要检查文本内容


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
    return True

#chunk size和overlap不确定，暂定数值
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
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

#使用sentence-transformers神经网络嵌入向量，计算余弦相似度，并使用最大相似度平均法
def compute_avg_max_similarity(brief_chunks: List[str], oral_chunks: List[str], model: SentenceTransformer) -> float:
    if not brief_chunks or not oral_chunks:
        return float('nan')
    
    # Encode chunks using neural embeddings
    print(f"  Encoding {len(brief_chunks)} brief chunks...")
    brief_embeddings = model.encode(brief_chunks, show_progress_bar=False, convert_to_numpy=True)
    
    print(f"  Encoding {len(oral_chunks)} oral chunks...")
    oral_embeddings = model.encode(oral_chunks, show_progress_bar=False, convert_to_numpy=True)
    
    # Compute pairwise cosine similarities: shape (n_brief, n_oral)
    sims = cosine_similarity(brief_embeddings, oral_embeddings)
    
    # Check for NaN values
    if np.isnan(sims).any():
        raise ValueError("Found NaN values in similarity matrix - this indicates data quality issues")
    
    # For each brief chunk, find the maximum similarity with any oral chunk
    max_sims_per_brief = np.max(sims, axis=1)  # shape: (n_brief,)
    
    # Return the average of these maximum similarities
    return float(np.mean(max_sims_per_brief))


def process_case(case_dir: Path, chunk_size: int, overlap: int, model: SentenceTransformer) -> Dict:
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

    # Collect party briefs (only those with valid text)
    brief_texts: List[Tuple[str, str]] = []  # (filename, text)
    for p in sorted(json_dir.glob('*.json')):
        if p.name.endswith('__corpus.json'):
            continue
        if not is_party_brief(p.name):
            continue
        t = extract_brief_text(p)
        if t and t.strip():
            brief_texts.append((p.name, t))

    brief_chunks = []
    brief_chunk_map = []  # (brief_filename, chunk_index)
    for fname, text in brief_texts:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, c in enumerate(chunks):
            brief_chunks.append(c)
            brief_chunk_map.append((fname, idx))

    try:
        avg_sim = compute_avg_max_similarity(brief_chunks, oral_chunks, model)
    except Exception as e:
        return {
            'case': case_dir.name,
            'error': f'similarity computation failed: {str(e)}',
            'num_brief_docs': len(brief_texts),
            'num_brief_chunks': len(brief_chunks),
            'num_oral_chunks': len(oral_chunks),
        }

    return {
        'case': case_dir.name,
        'num_brief_docs': len(brief_texts),
        'num_brief_chunks': len(brief_chunks),
        'num_oral_chunks': len(oral_chunks),
        'avg_brief_oral_cosine': avg_sim,
    }


def main():
    parser = argparse.ArgumentParser(description='Compute average brief-oral similarity for cases')
    parser.add_argument('--data-root', type=str, required=False,
                        default=str(Path(__file__).resolve().parents[1] / 'data'),
                        help='Root directory containing case folders (e.g., data)')
    parser.add_argument('--cases', type=str, nargs='*', default=None,
                        help='Case docket numbers to process (e.g., 17-773 17-21)')
    parser.add_argument('--all-cases', action='store_true',
                        help='Process all cases in the data directory')
    parser.add_argument('--chunk-size', type=int, default=800)
    parser.add_argument('--overlap', type=int, default=200)
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model name (default: all-MiniLM-L6-v2)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    
    # Determine which cases to process
    if args.all_cases:
        # Find all case directories
        case_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
        cases = [d.name for d in case_dirs]
        print(f"Found {len(cases)} cases in {data_root}")
        scope = "all_cases"
    elif args.cases:
        cases = args.cases
        scope = "_".join(cases[:3])  # Use first 3 case names in filename
        if len(cases) > 3:
            scope += f"_and_{len(cases)-3}_more"
    else:
        # Default to example cases
        cases = ['17-773', '17-21']
        scope = "example_cases"
        print("No cases specified, using default examples. Use --all-cases to process all cases.")
    
    print(f"Will process {len(cases)} case(s)\n")

    # Load the sentence-transformer model
    print(f"Loading embedding model: {args.model}...")
    model = SentenceTransformer(args.model)
    print("Model loaded successfully.\n")

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] Processing case: {case}")
        case_dir = data_root / case
        res = process_case(case_dir, chunk_size=args.chunk_size, overlap=args.overlap, model=model)
        results.append(res)
        print()

    # Generate output filename based on model and scope
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model name (e.g., "all-MiniLM-L6-v2" -> "minilm")
    model_short = args.model.lower().replace('all-', '').replace('-base', '').replace('-v2', '').replace('-l6', '')
    if 'mpnet' in model_short:
        model_short = 'mpnet'
    elif 'minilm' in model_short:
        model_short = 'minilm'
    else:
        model_short = model_short.split('-')[0][:10]  # Take first part, max 10 chars
    
    output_filename = f"{scope}_similarity_{model_short}.jsonl"
    out_path = output_dir / output_filename
    
    with out_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {out_path}")
    print(f"  Total cases processed: {len(results)}")
    print(f"  Model used: {args.model}")
    print(f"{'='*60}\n")

    # Also print a concise summary
    print("Summary:")
    for r in results:
        print(f"{r['case']}: docs={r.get('num_brief_docs')} brief_chunks={r.get('num_brief_chunks')} oral_chunks={r.get('num_oral_chunks')} avg={r.get('avg_brief_oral_cosine')}")


if __name__ == '__main__':
    main()


