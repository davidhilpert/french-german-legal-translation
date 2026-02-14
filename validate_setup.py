#!/usr/bin/env python3
"""
Setup validation script for French-German Legal Translation project.

Checks that all dependencies are installed correctly and the
preprocessing pipeline can run.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version is 3.8+."""
    print("Checking Python version...", end=" ")
    if sys.version_info >= (3, 8):
        print(f"✓ {sys.version.split()[0]}")
        return True
    else:
        print(f"✗ {sys.version.split()[0]} (need 3.8+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    import_name = import_name or package_name
    print(f"Checking {package_name}...", end=" ")
    try:
        __import__(import_name)
        print("✓")
        return True
    except ImportError:
        print("✗")
        return False


def check_spacy_models():
    """Check if spaCy language models are installed."""
    print("Checking spaCy language models...")

    try:
        import spacy

        # Check French model
        print("  - fr_core_news_sm...", end=" ")
        try:
            spacy.load("fr_core_news_sm")
            print("✓")
            fr_ok = True
        except OSError:
            print("✗ (run: python -m spacy download fr_core_news_sm)")
            fr_ok = False

        # Check German model
        print("  - de_core_news_sm...", end=" ")
        try:
            spacy.load("de_core_news_sm")
            print("✓")
            de_ok = True
        except OSError:
            print("✗ (run: python -m spacy download de_core_news_sm)")
            de_ok = False

        return fr_ok and de_ok
    except ImportError:
        print("  ✗ spaCy not installed")
        return False


def check_project_structure():
    """Check that project directories exist."""
    print("Checking project structure...")

    required_dirs = [
        "src/data",
        "data/raw",
        "data/processed",
        "tests",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        print(f"  - {dir_path}...", end=" ")
        if path.exists():
            print("✓")
        else:
            print("✗")
            all_exist = False

    return all_exist


def check_preprocessing_module():
    """Check that preprocessing module can be imported."""
    print("Checking preprocessing module...", end=" ")
    try:
        from src.data.preprocess import LegalTextPreprocessor
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ ({e})")
        return False


def test_preprocessing_sample():
    """Test preprocessing on sample text."""
    print("\nTesting preprocessing on sample text...")

    try:
        from src.data.preprocess import LegalTextPreprocessor

        print("  - Initializing preprocessor...", end=" ")
        preprocessor = LegalTextPreprocessor()
        print("✓")

        # Test text cleaning
        print("  - Testing text cleaning...", end=" ")
        test_text = "Article   123\\n\\n\\nSection  A"
        cleaned = preprocessor.clean_text(test_text)
        assert "Article 123" in cleaned
        print("✓")

        # Test French segmentation
        print("  - Testing French segmentation...", end=" ")
        fr_text = "Première phrase. Deuxième phrase."
        fr_sents = preprocessor.segment_sentences(fr_text, "fr")
        assert len(fr_sents) == 2
        print(f"✓ ({len(fr_sents)} sentences)")

        # Test German segmentation
        print("  - Testing German segmentation...", end=" ")
        de_text = "Erster Satz. Zweiter Satz."
        de_sents = preprocessor.segment_sentences(de_text, "de")
        assert len(de_sents) == 2
        print(f"✓ ({len(de_sents)} sentences)")

        # Test token counting
        print("  - Testing token counting...", end=" ")
        fr_count = preprocessor.count_tokens("Le système des montants.", "fr")
        de_count = preprocessor.count_tokens("Das System der Beträge.", "de")
        print(f"✓ (FR: {fr_count}, DE: {de_count} tokens)")

        return True

    except Exception as e:
        print(f"✗ ({e})")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("SETUP VALIDATION")
    print("=" * 60)
    print()

    checks = []

    # Core dependencies
    print("CORE DEPENDENCIES")
    print("-" * 60)
    checks.append(check_python_version())
    checks.append(check_package("torch"))
    checks.append(check_package("transformers"))
    checks.append(check_package("datasets"))
    checks.append(check_package("spacy"))
    checks.append(check_package("pandas"))
    checks.append(check_package("numpy"))
    print()

    # spaCy models
    print("SPACY LANGUAGE MODELS")
    print("-" * 60)
    checks.append(check_spacy_models())
    print()

    # Project structure
    print("PROJECT STRUCTURE")
    print("-" * 60)
    checks.append(check_project_structure())
    print()

    # Preprocessing module
    print("PREPROCESSING MODULE")
    print("-" * 60)
    checks.append(check_preprocessing_module())

    # Test preprocessing if module loaded
    if checks[-1]:
        checks.append(test_preprocessing_sample())

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print()
        print("Your setup is complete! Next steps:")
        print("  1. Download data: python -m src.data.download")
        print("  2. Preprocess data: python -m src.data.preprocess")
        print("  3. Run tests: pytest tests/")
        return 0
    else:
        print(f"✗ Some checks failed ({passed}/{total} passed)")
        print()
        print("Please fix the issues above and run this script again.")
        print("For help, see SETUP.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
