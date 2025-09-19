# USSC Spider

A Python crawler for downloading and processing legal briefs and replies from the US Supreme Court (SCOTUS) website.

## Features

- Downloads PDF documents from SCOTUS docket pages
- Filters for briefs and replies only (ignores other document types)
- Organizes files by case number in separate folders
- Generates descriptive filenames with date and document description
- Extracts text content from PDFs using multiple methods (pdfminer, PyMuPDF, OCR)
- Outputs structured JSON metadata for each document

## Requirements

```bash
pip install requests beautifulsoup4 lxml pymupdf
# Optional: for better text extraction
pip install pdfminer.six
# Optional: for OCR support
pip install ocrmypdf
```

## Usage

```bash
python crawler.py \
  --queries-json cases.jsonl \
  --output-dir ./data \
  --site scotus
```

### Arguments

- `--queries-json`: Input JSON/JSONL file containing case docket numbers
- `--output-dir`: Output directory for downloaded files (default: ./data)
- `--site`: Site key (currently only 'scotus' supported)
- `--base-url`: Base URL for the site (optional)
- `--user-agent`: User agent string (default: crawler-pdf-json/1.0)
- `--min-interval`: Minimum interval between requests in seconds (default: 1.0)
- `--enable-ocr`: Enable OCR for text extraction (0 or 1, default: 0)

## Input Format

The input JSON/JSONL file should contain records with `docket_no` field:

```json
{"docket_no": "17-130"}
{"docket_no": "18-123"}
```

## Output Structure

```
data/
├── 17-130/
│   ├── pdf/
│   │   ├── Dec132017_Reply of petitioners Raymond J. Lucia, et al. filed.pdf
│   │   └── Jan152018_Brief of respondent Securities and Exchange Commission filed.pdf
│   └── json/
│       ├── Dec132017_Reply of petitioners Raymond J. Lucia, et al. filed.json
│       └── Jan152018_Brief of respondent Securities and Exchange Commission filed.json
└── logs/
    └── app.log
```

## Document Filtering

The crawler only downloads documents that:
- Are marked as "Main Document" (not "Certificate of Word Count" or "Proof of Service")
- Contain "brief" or "reply" in their description
- Are from cases with docket numbers in format YY-#### (e.g., 17-130)

## License

MIT License