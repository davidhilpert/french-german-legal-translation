# Setup Guide

Complete setup instructions for the French-German Legal Translation project.

## Prerequisites

- Python 3.8 or higher
- R 4.0 or higher
- Docker (optional, recommended)
- Git with Git LFS (for large files)

## Quick Start with Docker

If you're using VS Code with Dev Containers (recommended):

```bash
# 1. Clone the repository
git clone https://github.com/[yourusername]/french-german-legal-translation.git
cd french-german-legal-translation

# 2. Open in VS Code
code .

# 3. Reopen in Dev Container
# Command Palette (Ctrl+Shift+P) -> "Dev Containers: Reopen in Container"

# 4. Install Python dependencies (inside container)
pip install -r requirements.txt

# 5. Install R packages (inside container)
R -e "install.packages('eurlex', repos='https://cloud.r-project.org/')"
```

## Manual Setup (Without Docker)

### Step 1: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install R

#### macOS (using Homebrew)

```bash
brew install r
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install r-base r-base-dev
```

#### Windows

Download and install from: https://cran.r-project.org/bin/windows/base/

### Step 3: Install R Packages

Open R and install the eurlex package:

```bash
R
```

In the R console:

```r
install.packages("eurlex")
library(eurlex)  # Test that it loads

# Optional: Install additional helpful packages
install.packages(c("dplyr", "tidyr"))

quit()  # Exit R
```

### Step 4: Install rpy2 (Python-R Bridge)

This should already be installed from requirements.txt, but if not:

```bash
pip install rpy2
```

#### Troubleshooting rpy2 on macOS

If you encounter errors with rpy2 on macOS:

```bash
# Set R_HOME environment variable
export R_HOME=/Library/Frameworks/R.framework/Resources

# Or add to your ~/.bashrc or ~/.zshrc:
echo 'export R_HOME=/Library/Frameworks/R.framework/Resources' >> ~/.zshrc
```

#### Troubleshooting rpy2 on Linux

```bash
# Install R development files
sudo apt-get install libcurl4-openssl-dev libssl-dev libxml2-dev

# Set R_HOME
export R_HOME=/usr/lib/R
```

### Step 5: Verify Installation

Test that everything is working:

```bash
# Test Python packages
python -c "import torch; import transformers; print('✓ PyTorch and Transformers OK')"

# Test R integration
python -c "import rpy2.robjects as ro; print('✓ rpy2 OK')"

# Test eurlex package
python -c "from rpy2.robjects.packages import importr; eurlex = importr('eurlex'); print('✓ eurlex OK')"
```

Expected output:
```
✓ PyTorch and Transformers OK
✓ rpy2 OK
✓ eurlex OK
```

## Configuration

### GPU Support (Optional, for Training)

If you have an NVIDIA GPU and want to use it for training:

#### Check CUDA availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Install PyTorch with CUDA support

Visit https://pytorch.org/get-started/locally/ and follow instructions for your system.

Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment Variables

Create a `.env` file in the project root for configuration:

```bash
# .env
DATA_DIR=data/raw
CACHE_DIR=cache
HF_TOKEN=your_huggingface_token_here  # Optional, for model uploads
```

## Testing Your Setup

### Step 1: Install spaCy Language Models

Before preprocessing, install the required spaCy models:

```bash
# Use the setup script
bash setup_spacy.sh

# Or install manually
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
```

### Step 2: Test Data Download (Small Sample)

```bash
# Download a small sample (10 documents) to test the pipeline
python src/data/download.py --limit 50 --max-documents 10

# Check the logs
tail -f data/download.log
```

Expected output:
- Files in `data/raw/french/` and `data/raw/german/`
- Metadata files in `data/raw/metadata/`
- Log file at `data/download.log`

### Step 3: Test Preprocessing Pipeline

After downloading data, test the preprocessing:

```bash
# Run preprocessing on downloaded documents
python -m src.data.preprocess

# Check the output
head -n 5 data/processed/parallel_sentences.jsonl
cat data/processed/preprocessing_stats.json
```

Expected output:
- `data/processed/parallel_sentences.jsonl` - Aligned sentence pairs
- `data/processed/preprocessing_stats.json` - Statistics
- Detailed logs showing processing progress

### Step 4: Run Unit Tests

```bash
# Test preprocessing module
pytest tests/test_preprocessing.py -v

# Run all tests
pytest tests/ -v
```

### Test R Integration

```python
# test_r_integration.py
from rpy2.robjects.packages import importr

# Load eurlex
eurlex = importr('eurlex')

# Test query (limit to 5 results)
query = eurlex.elx_make_query(resource_type='caselaw', limit=5)
print("✓ Query created successfully")

# Test execution
results = eurlex.elx_run_query(query)
print(f"✓ Query returned {len(results)} columns")

print("\nSetup test passed! You're ready to download data.")
```

Run with:
```bash
python test_r_integration.py
```

## Common Issues

### Issue 1: rpy2 cannot find R

**Error:** `RuntimeError: R_HOME not defined`

**Solution:**
```bash
# Find R installation
which R  # On Unix/Linux
where R  # On Windows

# Set R_HOME
export R_HOME=/path/to/R/installation
```

### Issue 2: eurlex package not found

**Error:** `RRuntimeError: package 'eurlex' is not installed`

**Solution:**
```bash
R -e "install.packages('eurlex', repos='https://cloud.r-project.org/')"
```

### Issue 3: SSL certificate errors

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution:**
```bash
# Python SSL certificates
pip install --upgrade certifi

# R SSL certificates
R -e "install.packages('openssl')"
```

### Issue 4: Memory errors during download

**Error:** `MemoryError` or slow performance

**Solution:**
- Reduce batch size with `--max-documents`
- Increase delay with `--delay 2.0`
- Download in smaller year ranges

### Issue 5: PyTorch CPU-only on macOS with M1/M2

**Solution:**
```bash
# Install PyTorch with MPS (Metal) support for Apple Silicon
pip install torch torchvision torchaudio
```

## Development Workflow

### Recommended Tools

- **IDE:** VS Code with Python and R extensions
- **Formatter:** Black (Python), styler (R)
- **Linter:** pylint, mypy
- **Testing:** pytest
- **Notebooks:** Jupyter Lab

### VS Code Extensions

Install these extensions for the best experience:

```
ms-python.python
ms-toolsai.jupyter
ikuyadeu.r
rdebugger.r-debugger
```

## Development Pipeline

### Phase 1: Data Download ✅

Download CJEU documents from EUR-Lex API:

```bash
python -m src.data.download
```

See `data/raw/README.md` for output documentation.

### Phase 2: Data Preprocessing ✅

Preprocess raw documents into aligned sentence pairs:

```bash
# First, install spaCy models
bash setup_spacy.sh

# Then run preprocessing
python -m src.data.preprocess
```

**What it does:**
- Cleans text (unicode normalization, whitespace)
- Segments into sentences using spaCy
- Aligns French-German sentence pairs 1:1
- Filters by length (5-512 tokens)
- Saves in JSON Lines format

See `data/processed/README.md` for output format documentation.

### Phase 3: Data Exploration (Next)

Explore the preprocessed data:

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

### Phase 4: Baseline Model (Coming Soon)

Evaluate Helsinki-NLP opus-mt-fr-de as baseline.

### Phase 5: Fine-tuning (Coming Soon)

Train mBART-large-50 on legal data.

## Getting Help

- **Documentation:** See README.md and CLAUDE.md
- **Issues:** Check data/download.log for errors
- **eurlex help:** https://michalovadek.github.io/eurlex/
- **rpy2 help:** https://rpy2.github.io/

## Contributing

Before contributing:

1. Run tests: `pytest tests/`
2. Format code: `black src/`
3. Check types: `mypy src/`
4. Lint: `pylint src/`

---

**Last updated:** February 14, 2026
