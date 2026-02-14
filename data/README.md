# Data Directory

This directory contains all data for the French-German legal translation project, including raw CJEU documents and preprocessed parallel corpora.

## Directory Structure

```
data/
├── raw/                    # Raw downloaded documents from EUR-Lex
│   ├── french/            # French source documents (*.txt)
│   ├── german/            # German target documents (*.txt)
│   └── metadata/          # Download metadata and alignment info
│       ├── download_metadata.csv      # Complete download statistics
│       └── alignment_index.json       # CELEX ID to file path mapping
│
├── processed/             # Preprocessed data ready for training
│   ├── train.json         # Training set (sentence pairs)
│   ├── validation.json    # Validation set
│   ├── test.json          # Test set
│   └── statistics.json    # Corpus statistics
│
└── download.log           # Download process log file
```

## Raw Data Format

### Document Files

Each CJEU ruling is saved as a plain text file named by its CELEX identifier:

- **French:** `data/raw/french/62019CJ0100.txt`
- **German:** `data/raw/german/62019CJ0100.txt`

**CELEX ID Format:** `6YYYYCJXXXX`
- `6` = Case law (sector code)
- `YYYY` = Year (e.g., 2019)
- `CJ` = Court of Justice
- `XXXX` = Case number

### Metadata Files

#### download_metadata.csv

Complete metadata for all downloaded documents:

| Column | Description |
|--------|-------------|
| `celex` | CELEX identifier |
| `work_date_document` | Document date |
| `work_date_lodged` | Case lodging date |
| `ecli` | European Case Law Identifier |
| `french_success` | Boolean: French download succeeded |
| `german_success` | Boolean: German download succeeded |
| `french_path` | Path to French document |
| `german_path` | Path to German document |
| `french_length` | Character count (French) |
| `german_length` | Character count (German) |
| `timestamp` | Download timestamp (ISO 8601) |

#### alignment_index.json

JSON mapping of CELEX IDs to aligned parallel documents:

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

## Processed Data Format

Preprocessed data is stored in JSON Lines format (one JSON object per line) for efficient loading:

```json
{
  "id": "62019CJ0100_sent_001",
  "celex_id": "62019CJ0100",
  "sentence_id": 1,
  "source": "Le présent renvoi préjudiciel porte sur l'interprétation...",
  "target": "Das vorliegende Vorabentscheidungsersuchen betrifft die Auslegung...",
  "source_length": 87,
  "target_length": 92,
  "split": "train"
}
```

### Fields

- **id:** Unique identifier (CELEX_ID + sentence number)
- **celex_id:** Source document CELEX identifier
- **sentence_id:** Sentence position in document
- **source:** French source sentence
- **target:** German target sentence
- **source_length:** Token count (French)
- **target_length:** Token count (German)
- **split:** Dataset split (train/validation/test)

## Data Statistics

Expected corpus size after preprocessing:

| Metric | Value |
|--------|-------|
| Documents | ~1,000-5,000 rulings |
| Sentence pairs | ~10,000-50,000 pairs |
| Avg. source length | 30-50 tokens |
| Avg. target length | 32-52 tokens |
| Train/Val/Test split | 80% / 10% / 10% |

## Data Provenance

**Source:** Court of Justice of the European Union (CJEU)
**Document Type:** Preliminary reference rulings (C-cases)
**Access Method:** EUR-Lex API via eurlex R package
**Languages:** French (source) → German (target)
**Translation Quality:** Professional human translations (gold standard)
**License:** Public domain (official EU documents)

## Citation

```bibtex
@Manual{eurlex,
  title = {eurlex: Retrieve Data on European Union Law},
  author = {Michal Ovádek},
  year = {2021},
  url = {https://CRAN.R-project.org/package=eurlex},
}
```

## Usage

### Download Raw Data

```bash
# Download all CJEU cases (takes several hours)
python src/data/download.py

# Download with date filtering
python src/data/download.py --start-year 2015 --end-year 2023

# Test with limited downloads
python src/data/download.py --limit 100 --max-documents 10
```

### Preprocess Data

```bash
# Preprocess and create train/val/test splits
python src/data/preprocess.py --input data/raw --output data/processed
```

## Quality Checks

After downloading, verify data integrity:

```python
import json
import pandas as pd

# Load metadata
metadata = pd.read_csv('data/raw/metadata/download_metadata.csv')

# Check success rate
success_rate = metadata[
    metadata['french_success'] & metadata['german_success']
].shape[0] / len(metadata)

print(f"Parallel document success rate: {success_rate:.1%}")

# Load alignment index
with open('data/raw/metadata/alignment_index.json') as f:
    alignment = json.load(f)

print(f"Aligned document pairs: {len(alignment)}")
```

## Known Issues

1. **Missing translations:** Some documents may not have both French and German versions
2. **Encoding:** Older documents may have encoding issues (handled by UTF-8 fallback)
3. **Rate limiting:** EUR-Lex API may throttle excessive requests (use `--delay` parameter)
4. **Document length:** Some rulings are very long (>10,000 tokens), requiring truncation

## Data Privacy

All data in this directory is:
- ✅ Public domain (official EU documents)
- ✅ Safe to version control (no sensitive information)
- ✅ Compliant with GDPR (no personal data)

**Note:** Files in `raw/` are large and should use Git LFS or be excluded from version control.

## Support

For issues with data download or processing:
1. Check `data/download.log` for error messages
2. Verify R and eurlex package installation: `R -e "library(eurlex)"`
3. Test EUR-Lex connectivity: `curl -I https://eur-lex.europa.eu/`
4. Consult eurlex documentation: https://michalovadek.github.io/eurlex/
