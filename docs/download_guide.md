# CJEU Document Download Guide

Quick reference for downloading CJEU preliminary reference rulings using the EUR-Lex API.

## Overview

The `src/data/download.py` script uses the [eurlex R package](https://michalovadek.github.io/eurlex/) via rpy2 to:

1. Query CJEU preliminary reference cases from EUR-Lex
2. Download parallel French-German documents
3. Save aligned documents with metadata
4. Generate alignment index for preprocessing

## Basic Usage

### Test Download (5 documents)

Start with a small test to verify everything works:

```bash
python src/data/download.py --limit 50 --max-documents 5
```

This will:
- Query up to 50 CJEU cases
- Download only the first 5 documents
- Take approximately 2-5 minutes

### Full Download (All Available Cases)

```bash
python src/data/download.py
```

⚠️ **Warning:** This may take several hours and download thousands of documents.

### Download by Date Range

```bash
# Download cases from 2015-2023
python src/data/download.py --start-year 2015 --end-year 2023

# Download recent cases (2020-present)
python src/data/download.py --start-year 2020
```

### Download with Custom Settings

```bash
# Slower rate limiting (more polite to EUR-Lex)
python src/data/download.py --delay 2.0

# Query limit + download limit
python src/data/download.py --limit 200 --max-documents 50

# Custom output directory
python src/data/download.py --output-dir /path/to/custom/data
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `data/raw` | Output directory for documents |
| `--start-year` | int | None | Starting year (e.g., 2000) |
| `--end-year` | int | None | Ending year (e.g., 2023) |
| `--limit` | int | None | Max cases to query (for testing) |
| `--max-documents` | int | None | Max documents to download (for testing) |
| `--delay` | float | 1.0 | Delay between API requests (seconds) |

## Understanding the Output

### Directory Structure Created

```
data/raw/
├── french/                    # French source documents
│   ├── 62019CJ0100.txt
│   ├── 62019CJ0102.txt
│   └── ...
├── german/                    # German target documents
│   ├── 62019CJ0100.txt
│   ├── 62019CJ0102.txt
│   └── ...
└── metadata/
    ├── download_metadata.csv     # Complete download statistics
    └── alignment_index.json      # CELEX ID → file path mapping
```

### Log File

All activity is logged to `data/download.log`:

```bash
# Watch download progress in real-time
tail -f data/download.log

# View recent errors
grep ERROR data/download.log | tail -20

# Count successful downloads
grep "✓" data/download.log | wc -l
```

### Download Metadata

The `download_metadata.csv` file contains:

```csv
celex,work_date_document,french_success,german_success,french_path,german_path,french_length,german_length,timestamp
62019CJ0100,2021-03-15,True,True,data/raw/french/62019CJ0100.txt,data/raw/german/62019CJ0100.txt,45230,48120,2026-02-14T10:30:45
```

### Alignment Index

The `alignment_index.json` maps CELEX IDs to file paths:

```json
{
  "62019CJ0100": {
    "french_path": "data/raw/french/62019CJ0100.txt",
    "german_path": "data/raw/german/62019CJ0100.txt",
    "french_length": 45230,
    "german_length": 48120,
    "timestamp": "2026-02-14T10:30:45"
  }
}
```

## Monitoring Progress

### Console Output

The script displays progress with tqdm:

```
Downloading: 100%|████████████| 50/50 [05:30<00:00, 6.6s/it]
```

### Summary Statistics

At completion, you'll see:

```
============================================================
DOWNLOAD SUMMARY
============================================================
Total cases processed: 50
Both languages downloaded: 45 (90.0%)
French only: 3
German only: 1
Both failed: 1
============================================================
```

## Troubleshooting

### Issue: No R_HOME environment variable

```bash
# macOS
export R_HOME=/Library/Frameworks/R.framework/Resources

# Linux
export R_HOME=/usr/lib/R

# Add to ~/.bashrc or ~/.zshrc to make permanent
```

### Issue: eurlex package not found

```bash
R -e "install.packages('eurlex', repos='https://cloud.r-project.org/')"
```

### Issue: Rate limiting / connection errors

Increase delay between requests:

```bash
python src/data/download.py --delay 3.0
```

### Issue: Some documents fail to download

This is normal - not all cases have both French and German versions. Check:

```python
import pandas as pd

metadata = pd.read_csv('data/raw/metadata/download_metadata.csv')

# Cases with both languages
both = metadata[metadata['french_success'] & metadata['german_success']]
print(f"Parallel documents: {len(both)}")

# Cases missing German
missing_de = metadata[metadata['french_success'] & ~metadata['german_success']]
print(f"Missing German: {len(missing_de)}")
```

### Issue: Memory errors

Download in smaller batches:

```bash
# Download 2015-2018
python src/data/download.py --start-year 2015 --end-year 2018

# Then download 2019-2023
python src/data/download.py --start-year 2019 --end-year 2023
```

## Python API Usage

You can also use the downloader programmatically:

```python
from src.data.download import EURLexDownloader

# Initialize downloader
downloader = EURLexDownloader(
    output_dir='data/raw',
    languages=('fr', 'de'),
    delay=1.0
)

# Query cases
cases_df = downloader.query_cjeu_cases(
    start_year=2020,
    end_year=2023,
    limit=100
)

print(f"Found {len(cases_df)} cases")

# Download first 10 for testing
results_df = downloader.download_parallel_documents(
    cases_df,
    max_documents=10
)

# Save metadata
downloader.save_metadata(cases_df, results_df)
downloader.create_alignment_index(results_df)
```

## Best Practices

### 1. Start Small

Always test with `--max-documents 5` first to verify your setup.

### 2. Use Date Ranges

Download by year ranges to make the process manageable:

```bash
for year in {2015..2023}; do
    python src/data/download.py --start-year $year --end-year $year
done
```

### 3. Monitor Logs

Keep `tail -f data/download.log` running in a separate terminal.

### 4. Be Polite to EUR-Lex

Use appropriate delays (1-2 seconds) to avoid overloading the server:

```bash
python src/data/download.py --delay 1.5
```

### 5. Backup Metadata

The metadata files are small - back them up after each download session:

```bash
cp data/raw/metadata/download_metadata.csv backups/metadata_$(date +%Y%m%d).csv
```

## Expected Timeline

| Task | Duration | Notes |
|------|----------|-------|
| Test download (5 docs) | 2-5 min | Verify setup |
| Small batch (50 docs) | 15-30 min | Initial testing |
| Year range (100-200 docs) | 1-2 hours | Typical batch |
| Full corpus (1000+ docs) | 4-8 hours | Run overnight |

## Next Steps

After downloading:

1. **Verify downloads:**
   ```bash
   ls -lh data/raw/french/ | wc -l
   ls -lh data/raw/german/ | wc -l
   ```

2. **Inspect sample document:**
   ```bash
   head -n 20 data/raw/french/$(ls data/raw/french/ | head -1)
   ```

3. **Explore in Python:**
   ```python
   import json
   with open('data/raw/metadata/alignment_index.json') as f:
       alignment = json.load(f)
   print(f"Aligned pairs: {len(alignment)}")
   ```

4. **Proceed to preprocessing:**
   ```bash
   python src/data/preprocess.py
   ```

## EUR-Lex API Details

The script uses these eurlex R functions:

- **`elx_make_query()`**: Creates SPARQL query for caselaw
- **`elx_run_query()`**: Executes query, returns DataFrame
- **`elx_fetch_data()`**: Downloads document text by CELEX ID

### CELEX ID Format

CJEU case CELEX IDs follow this pattern:

```
6 2019 CJ 0100
│  │    │   │
│  │    │   └─ Case number (sequential)
│  │    └───── Court: CJ (Court of Justice)
│  └────────── Year
└───────────── Sector: 6 (Case law)
```

Example: `62019CJ0100` = Court of Justice case 100 from 2019

## Resources

- **eurlex documentation:** https://michalovadek.github.io/eurlex/
- **EUR-Lex portal:** https://eur-lex.europa.eu/
- **CELEX documentation:** [EUR-Lex CELEX Guide](https://eur-lex.europa.eu/content/tools/TableOfSectors/types_of_documents_in_eurlex.html)

---

**Questions or issues?** Check `data/download.log` or open an issue on GitHub.
