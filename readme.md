# USSC Spider

A Python tool for downloading merits briefs from the US Supreme Court (SCOTUS) and oral-argument transcripts from Oyez, then analyzing semantic similarity between them using neural embeddings.

---

## Features

- **Auto-discovers** SCOTUS cases via the Oyez API — no manual case list needed
- **Only processes dockets with data in both databases** — skips any case missing SCOTUS briefs or an Oyez oral argument transcript
- Downloads only **merits briefs** (petitioner, respondent, amicus curiae) — motions, letters and certificates are excluded
- Generates **unique, descriptive filenames** (`{date}_{role}_{party}_{hash}.pdf`)
- Computes **semantic similarity** (Sentence-Transformers) between briefs and oral arguments

---

## Quick Start

### 1. Install Dependencies

```bash
pip install requests beautifulsoup4 lxml pymupdf pdfminer.six sentence-transformers
```

### 2. Scrape Briefs + Transcripts (Combined)

Use `combined_scraper.py` as the single entry point. It checks both SCOTUS and Oyez for each docket and only saves data when **both** sources have content.

```bash
# Single term (most common)
py scripts/combined_scraper.py --term 2023

# Year range (e.g. 2015–2023 inclusive)
py scripts/combined_scraper.py --term 2015 2023

# All terms ever (takes several hours)
py scripts/combined_scraper.py --all-cases

# Specific dockets
py scripts/combined_scraper.py --cases 22-915 21-369

# Re-download everything (overwrite existing files)
py scripts/combined_scraper.py --term 2023 --overwrite

# Skip very old cases with no PDFs
py scripts/combined_scraper.py --all-cases --min-year 2000
```

### 3. Analyze Similarity

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
└── {docket}/                    e.g. 22-915/
    ├── pdf/                     Merits brief PDFs
    │   └── 20230420_petitioner_Twitter_a3f2b1.pdf
    ├── json/                    Brief metadata + extracted text (JSON)
    │   └── 20230420_petitioner_Twitter_a3f2b1.json
    └── transcription/           Oyez oral argument transcript
        └── 22-915__corpus.json

output/
├── logs/combined_scraper.log
└── all_cases_similarity_*.jsonl
```

Only dockets that have **both** SCOTUS merits briefs and an Oyez transcript are written here.

---

## Combined Scraper (`scripts/combined_scraper.py`)

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--term YEAR [YEAR]` | — | One year or a range, e.g. `--term 2023` or `--term 2015 2023` |
| `--all-cases` | — | Auto-discover all cases via Oyez API |
| `--cases DOCKET ...` | — | Specific dockets, e.g. `22-915 21-369` |
| `--output-dir DIR` | `./data` | Root output directory |
| `--min-interval SEC` | `1.5` | Seconds between requests |
| `--overwrite` | `False` | Re-download and overwrite existing files |

### Skip Logic

| Condition | Action |
|-----------|--------|
| Docket not found on Oyez, or no `oral_argument_audio` | `SKIP (no transcript)` |
| Docket found on Oyez but SCOTUS has no merits briefs | `SKIP (no briefs)` |
| Both briefs and transcript exist | Download everything |

### Document Filtering

Only downloads documents that:
- ✅ Are a petitioner or respondent brief, or a petitioner/respondent reply
- ✅ Are from cases with `YY-####` docket format
- ❌ Excludes amicus/amici curiae briefs, motions, certificates, proof of service, letters

### Filename Format

```
{YYYYMMDD}_{role}_{party_name}_{hash6}.pdf
```
Example: `20221005_amicus_Mountain_States_Legal_Foundation_b7c3d2.pdf`

### End-of-run Summary

```
DONE  total=N  both_found=N  skip_no_transcript=N  skip_no_briefs=N  failed=N
```

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
- Logs are written to `output/logs/combined_scraper.log`
- First similarity run downloads the model (~90 MB for default, ~420 MB for MPNet)

---

## License

MIT License