#!/bin/bash
# Setup script for downloading spaCy language models
# Run this after installing requirements.txt

echo "Downloading spaCy language models for French and German..."
echo "This may take a few minutes..."

# Download French model
echo "Downloading French model (fr_core_news_sm)..."
python -m spacy download fr_core_news_sm

# Download German model
echo "Downloading German model (de_core_news_sm)..."
python -m spacy download de_core_news_sm

echo "spaCy models installed successfully!"
echo "You can now run the preprocessing pipeline:"
echo "  python -m src.data.preprocess"
