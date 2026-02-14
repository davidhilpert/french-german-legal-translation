# Installation Guide

## Quick Start - Install Now (Current Container)

### Option 1: Run the Installation Script (Easiest)

```bash
bash install_dependencies.sh
```

This script will:
1. Install R and required system libraries
2. Install the eurlex R package
3. Install Python packages from requirements.txt
4. Verify the installation

**Time:** ~5-10 minutes

### Option 2: Manual Installation (Step by Step)

If you prefer to run commands manually or if the script fails:

```bash
# 1. Install R and system libraries
sudo apt-get update
sudo apt-get install -y \
  r-base \
  r-base-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  libxml2-dev \
  libfontconfig1-dev \
  libharfbuzz-dev \
  libfribidi-dev \
  libfreetype6-dev \
  libpng-dev \
  libtiff5-dev \
  libjpeg-dev

# 2. Install eurlex R package
R -e "install.packages('eurlex', repos='https://cloud.r-project.org/', dependencies=TRUE)"

# 3. Install Python packages
pip install --no-cache-dir -r requirements.txt

# 4. Verify installation
python test_setup.py
```

### Verify Installation

After installation, run:

```bash
# Comprehensive test
python test_setup.py

# Quick checks
python -c "import rpy2; print('✓ rpy2 OK')"
python -c "import torch; print('✓ PyTorch OK')"
python -c "import transformers; print('✓ Transformers OK')"
R -e "library(eurlex); cat('✓ eurlex OK\n')"
```

Expected output: All checks should pass ✓

### Test Data Download

Once everything is installed, test the download script:

```bash
# Small test (5 documents, ~3 minutes)
python src/data/download.py --limit 50 --max-documents 5

# Check the results
ls -lh data/raw/french/
ls -lh data/raw/german/
cat data/download.log | tail -20
```

---

## Permanent Setup (Future Container Rebuilds)

The `.devcontainer/devcontainer.json` file has been updated to include R permanently.

### What Changed

**Added R Feature:**
```json
"ghcr.io/rocker-org/devcontainer-features/r-apt:latest": {
  "version": "latest"
}
```

**Added R VS Code Extensions:**
- `ikuyadeu.r` - R language support
- `rdebugger.r-debugger` - R debugging

**Updated postCreateCommand:**
Now automatically installs Python packages AND eurlex R package when container is created.

### How to Rebuild Container

If you want to start fresh with the permanent setup:

1. **In VS Code:**
   - Command Palette (Ctrl+Shift+P or Cmd+Shift+P)
   - Select: "Dev Containers: Rebuild Container"
   - Wait 5-10 minutes for rebuild

2. **Verify after rebuild:**
   ```bash
   python test_setup.py
   ```

### When to Rebuild

You should rebuild the container if:
- You want a clean environment
- The current container has issues
- You've updated devcontainer.json with new features
- You're sharing this project with others (they'll get the full setup automatically)

---

## Troubleshooting

### Issue: R installation fails

```bash
# Check if you have sudo access
sudo echo "sudo works"

# If sudo doesn't work, you may need to install from a pre-built container
# Consider using: rocker/r-ver:4.3
```

### Issue: eurlex installation fails in R

```bash
# Try installing dependencies separately
R -e "install.packages(c('httr', 'xml2', 'dplyr', 'tidyr'), repos='https://cloud.r-project.org/')"
R -e "install.packages('eurlex', repos='https://cloud.r-project.org/')"
```

### Issue: rpy2 installation fails

```bash
# Install R development files
sudo apt-get install -y r-base-dev

# Set R_HOME environment variable
export R_HOME=/usr/lib/R
echo 'export R_HOME=/usr/lib/R' >> ~/.bashrc

# Reinstall rpy2
pip install --no-cache-dir rpy2
```

### Issue: PyTorch installation is slow

PyTorch is large (~2GB). If installation is very slow:

```bash
# Install CPU version (smaller, faster)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Out of disk space

Check available space:
```bash
df -h /workspaces
```

If low, clean up:
```bash
# Remove pip cache
rm -rf ~/.cache/pip

# Remove old Python packages
pip cache purge
```

---

## What Gets Installed

### System Packages (via apt)
- `r-base` - R language
- `r-base-dev` - R development files
- Various `-dev` libraries for R packages (curl, ssl, xml, etc.)

### R Packages
- `eurlex` - EUR-Lex API client
- Dependencies: httr, xml2, dplyr, tidyr, etc.

### Python Packages (from requirements.txt)

**Core ML/NLP:**
- torch>=2.0.0 (~2GB)
- transformers>=4.30.0
- datasets>=2.14.0
- sentencepiece, sacrebleu, sacremoses

**Data handling:**
- pandas, numpy
- requests, beautifulsoup4, lxml

**R integration:**
- rpy2>=3.5.0

**Development:**
- jupyter, black, pylint, pytest
- tqdm (progress bars)

**Total size:** ~3-4 GB

---

## Disk Space Requirements

| Component | Size |
|-----------|------|
| R + libraries | ~500 MB |
| PyTorch | ~2 GB |
| Other Python packages | ~1.5 GB |
| **Total** | **~4 GB** |

Make sure you have at least **5-6 GB free** before installing.

---

## Next Steps After Installation

1. **Verify setup:**
   ```bash
   python test_setup.py
   ```

2. **Test download (5 documents):**
   ```bash
   python src/data/download.py --limit 50 --max-documents 5
   ```

3. **Explore data:**
   ```bash
   jupyter notebook notebooks/
   ```

4. **Read documentation:**
   - `README.md` - Project overview
   - `CLAUDE.md` - Detailed project plan
   - `data/README.md` - Data format documentation
   - `docs/download_guide.md` - Download script reference

---

## FAQ

**Q: Do I need GPU for data download?**
A: No, GPU is only needed for model training (Phase 4). Data download works fine on CPU.

**Q: How long does installation take?**
A: 5-10 minutes for the installation script, plus 5-10 minutes if rebuilding the container.

**Q: Can I use a different R version?**
A: Yes, but eurlex requires R >= 4.0. The devcontainer uses the latest stable version.

**Q: What if I don't want to use R?**
A: You could access the EUR-Lex SPARQL endpoint directly with Python, but eurlex simplifies this significantly. Alternatively, you could download documents manually.

**Q: Is this setup permanent?**
A: Installation in the current container is temporary (lost on rebuild). The devcontainer.json updates are permanent (persist across rebuilds).

---

**Last updated:** February 14, 2026
