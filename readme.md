# USSC Spider

A Python tool for downloading legal briefs from the US Supreme Court and analyzing similarity between briefs and oral arguments using neural embeddings.

## Features

- 📥 Downloads PDF briefs and replies from SCOTUS docket pages
- 📄 Extracts text content from PDFs (supports OCR for scanned documents)
- 🧠 Computes semantic similarity using neural embeddings (Sentence-Transformers)
- 📊 Outputs structured JSON metadata and similarity scores

---

## Quick Start

### 1. Install Dependencies

```bash
# For web scraping
pip install requests beautifulsoup4 lxml pymupdf pdfminer.six

# For similarity analysis
pip install sentence-transformers

# Optional: for OCR support
pip install ocrmypdf
```

### 2. Download Briefs

Create an input file `cases.jsonl`:
```json
{"docket_no": "17-773"}
{"docket_no": "17-21"}
```

Run the scraper:
```bash
python3 scripts/crawler.py \
  --queries-json cases.jsonl \
  --output-dir ./data \
  --site scotus
```

### 3. Analyze Similarity

```bash
# Analyze all cases
python3 scripts/process_similarity.py --all-cases --output "results.jsonl"

# Or specific cases
python3 scripts/process_similarity.py --cases 17-773 17-21 --output "results.jsonl"
```

---

## Crawler Usage

### Basic Command
```bash
python3 scripts/crawler.py \
  --queries-json cases.jsonl \
  --output-dir ./data \
  --site scotus
```

### Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--queries-json` | - | Input JSONL file with docket numbers (required) |
| `--output-dir` | `./data` | Output directory |
| `--site` | - | Site key (use `scotus`) |
| `--min-interval` | `1.0` | Seconds between requests |
| `--enable-ocr` | `0` | Enable OCR (0=off, 1=on) |

### Output Structure
```
data/
├── 17-773/
│   ├── pdf/
│   │   └── Brief_Petitioner.pdf
│   └── json/
│       ├── Brief_Petitioner.json
│       └── 17-773__corpus.json
output/
├── all_cases_similarity_mpnet.jsonl  # Similarity analysis results
└── scraper_output.log                # Crawler logs
```

**Note:** All generated output files (similarity results, logs) are stored in the `output/` folder.

---

## Similarity Analysis

### What It Does
Computes semantic similarity between legal briefs and oral arguments using:
- **Neural embeddings** (Sentence-Transformers) instead of TF-IDF
- **Max-similarity averaging**: For each brief chunk, finds the best match with oral chunks, then averages

### Quick Commands

```bash
# All cases (recommended)
python3 scripts/process_similarity.py --all-cases --output "all_results.jsonl"

# Specific cases
python3 scripts/process_similarity.py --cases 17-773 17-21 --output "results.jsonl"

# High-quality model (slower but more accurate)
python3 scripts/process_similarity.py --all-cases --model "all-mpnet-base-v2" --output "results.jsonl"

# Custom chunk size
python3 scripts/process_similarity.py --all-cases --chunk-size 1500 --overlap 300 --output "results.jsonl"
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--all-cases` | - | Process all cases in data directory |
| `--cases` | - | Specific case numbers to process |
| `--model` | `all-MiniLM-L6-v2` | Embedding model name |
| `--chunk-size` | `1200` | Text chunk size (words) |
| `--overlap` | `200` | Chunk overlap (words) |
| `--output` | `similarity_results.jsonl` | Output file path |

### Available Models
| Model | Speed | Quality | Dimensions | Use Case |
|-------|-------|---------|------------|----------|
| `all-MiniLM-L6-v2` (default) | ⚡⚡⚡ | ⭐⭐⭐ | 384 | Fast testing, large batches |
| `all-mpnet-base-v2` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 768 | Best accuracy, final results |
| `paraphrase-MiniLM-L6-v2` | ⚡⚡⚡ | ⭐⭐⭐ | 384 | Paraphrase detection |

### Output Format
```json
{
  "case": "17-773",
  "num_brief_docs": 5,
  "num_brief_chunks": 234,
  "num_oral_chunks": 156,
  "avg_brief_oral_cosine": 0.6234,
  "empty_brief_samples": []
}
```

**Key metric:** `avg_brief_oral_cosine` (0-1, higher = more similar)

### Expected Runtime
- **Default model** (`all-MiniLM-L6-v2`):
  - Single case: ~10-30 seconds
  - All 188 cases: ~30-90 minutes
  
- **High-quality model** (`all-mpnet-base-v2`):
  - Single case: ~20-60 seconds
  - All 188 cases: ~60-120 minutes

---

## How It Works

### Similarity Calculation Method

**Vectorization:**
- Neural embeddings 

**Similarity Metric:**
```
For each brief chunk i:
  Find max similarity with all oral chunks j
Average these max similarities
```

This approach is more robust than simple averaging, focusing on best matches rather than being diluted by irrelevant content.

---

## Tips & Troubleshooting

### Crawler Tips
- Use `--min-interval 2.0` for large batches to be respectful of the server
- Check `output/scraper_output.log` for detailed logs
- The scraper skips cases before 2001 and "A-number" dockets

### Similarity Analysis Tips
- First run downloads the model (~90MB for default, ~420MB for MPNet)
- Use `--chunk-size 800` if you encounter memory errors
- Failed cases will have an `"error"` field in the output

### Common Issues

**ModuleNotFoundError: sentence_transformers**
```bash
pip install sentence-transformers
```

**No briefs found for case**
- Case may be pre-2001, an emergency application, or not yet at merits stage

**Empty text extraction**
```bash
python3 scripts/crawler.py --queries-json cases.jsonl --output-dir ./data --site scotus --enable-ocr 1
```

---

## Complete Workflow Example

```bash
# 1. Create input file
cat > my_cases.jsonl << EOF
{"docket_no": "17-773"}
{"docket_no": "17-21"}
EOF

# 2. Download briefs
python3 scripts/crawler.py \
  --queries-json my_cases.jsonl \
  --output-dir ./data \
  --site scotus

# 3. Analyze similarity
python3 scripts/process_similarity.py \
  --all-cases \
  --output "similarity_results.jsonl"

# 4. Check results
cat similarity_results.jsonl | python3 -m json.tool
```

---

## Document Filtering

The crawler only downloads documents that:
- ✅ Are marked as "Main Document"
- ✅ Contain "brief" or "reply" in description
- ✅ Are from cases with YY-#### format (e.g., 17-130)
- ❌ Skips certificates, proof of service, amicus briefs, motions

---

## License

MIT License