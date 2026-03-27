# USSC Spider

A Python tool for downloading merits briefs from the US Supreme Court (SCOTUS) and oral-argument transcripts from Oyez, then analyzing semantic similarity between them using neural embeddings.

---

## Features

- **Auto-discovers** all SCOTUS cases via the Oyez API — no manual case list needed
- Downloads only **merits briefs** (petitioner, respondent, amicus curiae) — motions, letters and certificates are excluded
- Generates **unique, descriptive filenames** (`{date}_{role}_{party}_{hash}.pdf`)
- Fetches **oral argument transcripts** from Oyez for all terms
- Computes **semantic similarity** (Sentence-Transformers) between briefs and oral arguments

---

## Quick Start

### 1. Install Dependencies

```bash
pip install requests beautifulsoup4 lxml pymupdf pdfminer.six sentence-transformers
```

### 2. Download Briefs — All Cases (Auto-Discovery)

```bash
# All cases, all terms (takes several hours)
py scripts/crawler.py --all-cases

# Single term
py scripts/crawler.py --all-cases --term 2023

# Year range  (e.g. 1997–2003 inclusive)
py scripts/crawler.py --all-cases --term 1997 2003

# Using a manual JSONL list instead
py scripts/crawler.py --queries-json cases.jsonl
```

### 3. Fetch Oral Argument Transcripts (Oyez)

```bash
# All cases, all terms
py scripts/oyez_scraper.py --all-cases

# Single term
py scripts/oyez_scraper.py --term 2023

# Year range
py scripts/oyez_scraper.py --term 1997 2003

# Specific dockets
py scripts/oyez_scraper.py --cases 22-300 21-476
```

### 4. Analyze Similarity

```bash
# All cases
py scripts/process_similarity.py --all-cases --output similarity_results.jsonl

# Specific cases
py scripts/process_similarity.py --cases 22-300 21-476 --output results.jsonl

# Higher-quality model (slower)
py scripts/process_similarity.py --all-cases --model all-mpnet-base-v2 --output results.jsonl
```

---

## Output Structure

```
data/
└── {docket}/               e.g. 22-300/
    ├── pdf/                Merits brief PDFs
    │   └── 20221005_petitioner_New_York_State_Rifle_a3f2b1.pdf
    ├── json/               Brief metadata + extracted text (JSON)
    │   └── 20221005_petitioner_New_York_State_Rifle_a3f2b1.json
    └── transcription/      Oyez oral argument transcript
        └── 22-300__corpus.json

output/
├── logs/scraper_output.log
└── all_cases_similarity_*.jsonl
```

---

## Crawler (`scripts/crawler.py`)

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--all-cases` | — | Auto-discover cases via Oyez API |
| `--term YEAR [YEAR]` | — | Limit to one year or a range, e.g. `--term 2023` or `--term 1997 2003` |
| `--queries-json FILE` | — | JSONL file with `docket_no` fields (alternative to `--all-cases`) |
| `--output-dir DIR` | `./data` | Root output directory |
| `--min-interval SEC` | `1.5` | Seconds between requests |
| `--min-year YEAR` | `1900` | Skip cases before this year (e.g. `--min-year 2000`) |

### Document Filtering

Only downloads documents that:
- ✅ Are a petitioner, respondent, or amicus/amici curiae brief
- ✅ Are from cases with `YY-####` docket format
- ❌ Excludes motions, certificates, proof of service, letters

### Filename Format

```
{YYYYMMDD}_{role}_{party_name}_{hash6}.pdf
```
Example: `20221005_amicus_Mountain_States_Legal_Foundation_b7c3d2.pdf`

---

## Oyez Scraper (`scripts/oyez_scraper.py`)

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--all-cases` | — | Fetch all cases from Oyez API (paginated) |
| `--term YEAR [YEAR]` | — | One year or range, e.g. `--term 2023` or `--term 1997 2003` |
| `--cases DOCKET ...` | — | Specific dockets, e.g. `22-300 21-476` |
| `--data-dir DIR` | `./data` | Root data directory |
| `--min-interval SEC` | `1.5` | Seconds between API requests |
| `--overwrite` | `False` | Overwrite existing transcript files |

Transcripts are saved to `data/{docket}/transcription/{docket}__corpus.json`.

---

## Similarity Analysis (`scripts/process_similarity.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--all-cases` | — | Process all cases in data directory |
| `--cases` | — | Specific dockets |
| `--model` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `--chunk-size` | `1200` | Text chunk size (words) |
| `--overlap` | `200` | Chunk overlap (words) |
| `--output` | `similarity_results.jsonl` | Output file |

**Output metric:** `avg_brief_oral_cosine` (0–1, higher = more similar)

---

## Tips & Troubleshooting

- Use `--min-interval 2.0` for large batches to be polite to servers
- Use `--min-year 2000` on the crawler to skip very old cases that have no PDFs
- Logs are written to `output/logs/scraper_output.log`
- First similarity run downloads the model (~90 MB for default, ~420 MB for MPNet)

---

## License

MIT License