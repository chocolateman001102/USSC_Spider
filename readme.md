Crawler v2：基于 SCOTUS Docket 的 PDF → JSON 抓取

变更要点（相对“初版”）：
- 新增站点适配器：ScotusDocketAdapter（无需提交搜索表单，直接命中 docket 页面）
- 支持从 JSON/JSONL 读取任务（字段至少包含 docket_no；可选 year）
- 将 Oyez/自有数据中的 docket_no 映射为 SCOTUS 的 docket code（如 21-471、24A949 → 24a949.html）
- 仍然输出统一 JSON Schema，不再生成 TXT

运行示例
    # JSONL/JSON 输入（推荐）
    python crawler_pdf_to_json.py \
      --queries-json cases.jsonl \
      --output-dir ./data \
      --site scotus

    # CSV 兼容模式（第一列是 query_code，例如 21-471 或 24A949）
    python crawler_pdf_to_json.py \
      --queries-file queries.csv \
      --output-dir ./data \
      --site scotus

依赖（requirements.txt）
    requests>=2.31.0
    beautifulsoup4>=4.12.3
    lxml>=5.2.2
    pdfminer.six>=20231228
    pymupdf>=1.24.7           # 可选：用于页数统计
    chardet>=5.2.0
    tenacity>=8.2.3
    tqdm>=4.66.4
    pydantic>=2.8.2
    # 可选：Playwright
    playwright>=1.45.0
    # 可选：OCR（本机需安装 ocrmypdf / tesseract 可执行文件）

输出目录结构
    {output-dir}/pdf/{yyyy}/{mm}/{query}_{hash8}.pdf
    {output-dir}/json/{yyyy}/{mm}/{query}_{hash8}.json
    {output-dir}/logs/app.log
