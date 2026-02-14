#!/bin/bash
# Installation script for R, eurlex, and Python dependencies
# Run with: bash install_dependencies.sh

set -e  # Exit on error

echo "=========================================="
echo "French-German Legal Translation Project"
echo "Dependency Installation Script"
echo "=========================================="
echo ""

# Step 1: Install R and system libraries
echo "Step 1/4: Installing R and system libraries..."
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

echo "✓ R installed successfully"
echo ""

# Step 2: Verify R installation
echo "Step 2/4: Verifying R installation..."
R --version | head -n 1
echo ""

# Step 3: Install eurlex R package
echo "Step 3/4: Installing eurlex R package..."
R -e "install.packages('eurlex', repos='https://cloud.r-project.org/', dependencies=TRUE)"
echo "✓ eurlex package installed"
echo ""

# Step 4: Install Python packages
echo "Step 4/4: Installing Python packages from requirements.txt..."
pip install --no-cache-dir -r requirements.txt
echo "✓ Python packages installed"
echo ""

# Verification
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

echo "Python packages:"
pip list | grep -E "(torch|transformers|rpy2|tqdm|pandas)" || echo "Some packages missing"
echo ""

echo "R packages:"
R -e "library(eurlex); cat('✓ eurlex loaded successfully\n')" 2>/dev/null || echo "✗ eurlex not available"
echo ""

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python test_setup.py"
echo "  2. Test download: python src/data/download.py --limit 50 --max-documents 5"
echo ""
