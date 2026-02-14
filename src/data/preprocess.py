"""
Preprocessing pipeline for French-German legal translation dataset.

This module handles paragraph extraction, alignment, cleaning, and filtering
of CJEU preliminary reference rulings (2000s onwards) for neural machine translation.

Modern CJEU judgments use numbered paragraphs (e.g., "11 Le quatrième..." in French
matches "11 Das vierte..." in German), enabling precise paragraph-level alignment.

The pipeline:
    1. Loads French-German document pairs from data/raw/
    2. Extracts numbered paragraphs from both documents
    3. Aligns paragraphs by matching paragraph numbers
    4. Cleans text (normalize whitespace, remove special chars)
    5. Filters by length constraints (5-512 tokens)
    6. Saves to data/processed/ in JSON Lines format

Example:
    $ python -m src.data.preprocess
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import unicodedata

try:
    import spacy
    from spacy.language import Language
except ImportError:
    raise ImportError(
        "spaCy is required for preprocessing. Install with: "
        "pip install spacy && "
        "python -m spacy download fr_core_news_sm && "
        "python -m spacy download de_core_news_sm"
    )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SentencePair:
    """Represents an aligned French-German sentence pair.

    Attributes:
        id: Unique identifier (celex_id + sentence index)
        source_text: French source sentence
        target_text: German target sentence
        celex_id: CELEX identifier for the document
        source_tokens: Number of tokens in source sentence
        target_tokens: Number of tokens in target sentence
        alignment_method: Method used for alignment ('numbered_paragraph' or 'position_based')
        paragraph_number: Paragraph number if using numbered_paragraph alignment
    """
    id: str
    source_text: str
    target_text: str
    celex_id: str
    source_tokens: int
    target_tokens: int
    alignment_method: str
    paragraph_number: Optional[int] = None


class LegalTextPreprocessor:
    """Preprocessor for modern CJEU legal documents (2000s onwards).

    Extracts numbered paragraphs from French-German parallel legal documents
    and aligns them by matching paragraph numbers.

    Attributes:
        nlp_fr: spaCy French language model (for tokenization)
        nlp_de: spaCy German language model (for tokenization)
        min_tokens: Minimum paragraph length in tokens
        max_tokens: Maximum paragraph length in tokens
        stats: Dictionary tracking preprocessing statistics
    """

    def __init__(
        self,
        min_tokens: int = 5,
        max_tokens: int = 512,
        spacy_model_fr: str = "fr_core_news_sm",
        spacy_model_de: str = "de_core_news_sm",
    ):
        """Initialize the preprocessor with language models.

        Args:
            min_tokens: Minimum sentence length (tokens) to keep
            max_tokens: Maximum sentence length (tokens) to keep
            spacy_model_fr: Name of French spaCy model
            spacy_model_de: Name of German spaCy model

        Raises:
            OSError: If spaCy models are not installed
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        logger.info("Loading spaCy language models...")
        try:
            self.nlp_fr: Language = spacy.load(spacy_model_fr)
            self.nlp_de: Language = spacy.load(spacy_model_de)
        except OSError as e:
            logger.error(
                f"Failed to load spaCy models. Please run:\n"
                f"  python -m spacy download {spacy_model_fr}\n"
                f"  python -m spacy download {spacy_model_de}"
            )
            raise e

        # Configure spaCy for efficiency (disable unused components)
        # We only need tokenization, not parsing, NER, or lemmatization
        self.nlp_fr.disable_pipes(["parser", "ner", "lemmatizer"])
        self.nlp_de.disable_pipes(["parser", "ner", "lemmatizer"])

        # Statistics tracking
        self.stats = {
            "documents_processed": 0,
            "documents_skipped_no_numbering": 0,
            "total_paragraph_pairs": 0,
            "filtered_too_short": 0,
            "filtered_too_long": 0,
            "final_paragraph_pairs": 0,
        }

        logger.info("Preprocessor initialized successfully")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Removes excessive whitespace, normalizes unicode characters,
        and handles special legal document formatting.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string

        Example:
            >>> preprocessor.clean_text("Article   123\\n\\n\\nSection  A")
            'Article 123 Section A'
        """
        # Normalize unicode characters (e.g., different types of spaces)
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters except newlines and tabs
        text = "".join(
            char for char in text
            if unicodedata.category(char)[0] != "C" or char in "\n\t"
        )

        # Replace multiple whitespaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Handle common legal document artifacts
        # Remove page numbers and headers (lines with only numbers and basic text)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely headers/artifacts
            if len(line) > 15 or any(c.isalnum() for c in line):
                cleaned_lines.append(line)

        text = " ".join(cleaned_lines)

        return text

    def extract_numbered_paragraphs(self, text: str) -> Optional[Dict[int, str]]:
        """Extract numbered paragraphs from CJEU judgment text.

        Modern CJEU judgments (2000s onwards) have paragraphs numbered sequentially
        (e.g., "11 Le quatrième..." in French, "11 Das vierte..." in German).
        This method extracts these numbered paragraphs for alignment.

        Args:
            text: Document text to parse

        Returns:
            Dictionary mapping paragraph numbers to paragraph text, or None if
            no numbered paragraphs are found (indicating an older document format)

        Example:
            >>> text = "11 Le quatrième...\\n12 La cinquième..."
            >>> preprocessor.extract_numbered_paragraphs(text)
            {11: "Le quatrième...", 12: "La cinquième..."}
        """
        # Pattern: line starts with digits followed by space
        # Capture the number and the rest of the paragraph
        paragraph_pattern = re.compile(r'^(\d+)\s+(.+)', re.MULTILINE)

        matches = paragraph_pattern.findall(text)

        # If we found fewer than 3 numbered paragraphs, assume this is not
        # a numbered document (could be headers, page numbers, etc.)
        if len(matches) < 3:
            return None

        # Build dictionary of paragraph number -> text
        numbered_paragraphs = {}
        for num_str, content in matches:
            para_num = int(num_str)
            # Only keep if numbers are sequential-ish (allow some gaps)
            # This filters out random numbers that aren't paragraph numbering
            if not numbered_paragraphs or para_num <= max(numbered_paragraphs.keys()) + 10:
                numbered_paragraphs[para_num] = content.strip()

        # Verify we have a reasonable sequence (at least 3 paragraphs)
        if len(numbered_paragraphs) < 3:
            return None

        return numbered_paragraphs

    def align_numbered_paragraphs(
        self,
        fr_paragraphs: Dict[int, str],
        de_paragraphs: Dict[int, str],
        celex_id: str,
    ) -> List[SentencePair]:
        """Align French and German paragraphs by paragraph number.

        Modern CJEU judgments have numbered paragraphs that can be directly
        matched between languages for precise alignment.

        Args:
            fr_paragraphs: French paragraphs keyed by paragraph number
            de_paragraphs: German paragraphs keyed by paragraph number
            celex_id: CELEX document identifier

        Returns:
            List of aligned sentence pairs with paragraph-level alignment
        """
        aligned_pairs = []

        # Find common paragraph numbers
        fr_nums = set(fr_paragraphs.keys())
        de_nums = set(de_paragraphs.keys())
        common_nums = sorted(fr_nums & de_nums)

        if not common_nums:
            logger.warning(f"{celex_id}: No common paragraph numbers found")
            return aligned_pairs

        # Log alignment quality
        fr_only = fr_nums - de_nums
        de_only = de_nums - fr_nums
        if fr_only or de_only:
            logger.debug(
                f"{celex_id}: Paragraph number mismatch - "
                f"FR only: {sorted(fr_only)[:5]}{'...' if len(fr_only) > 5 else ''}, "
                f"DE only: {sorted(de_only)[:5]}{'...' if len(de_only) > 5 else ''}"
            )

        # Align paragraphs by number
        for para_num in common_nums:
            fr_text = fr_paragraphs[para_num]
            de_text = de_paragraphs[para_num]

            # Count tokens
            fr_tokens = self.count_tokens(fr_text, "fr")
            de_tokens = self.count_tokens(de_text, "de")

            # Filter by length
            if fr_tokens < self.min_tokens or de_tokens < self.min_tokens:
                self.stats["filtered_too_short"] += 1
                continue

            if fr_tokens > self.max_tokens or de_tokens > self.max_tokens:
                self.stats["filtered_too_long"] += 1
                continue

            # Create sentence pair
            pair = SentencePair(
                id=f"{celex_id}_para{para_num:04d}",
                source_text=fr_text,
                target_text=de_text,
                celex_id=celex_id,
                source_tokens=fr_tokens,
                target_tokens=de_tokens,
                alignment_method="numbered_paragraph",
                paragraph_number=para_num,
            )

            aligned_pairs.append(pair)
            self.stats["total_paragraph_pairs"] += 1

        logger.info(
            f"{celex_id}: Aligned {len(aligned_pairs)} numbered paragraphs "
            f"(out of {len(common_nums)} common)"
        )

        return aligned_pairs

    def count_tokens(self, text: str, language: str) -> int:
        """Count tokens in text using spaCy tokenizer.

        Args:
            text: Text to tokenize
            language: Language code ('fr' or 'de')

        Returns:
            Number of tokens
        """
        if language == "fr":
            nlp = self.nlp_fr
        elif language == "de":
            nlp = self.nlp_de
        else:
            raise ValueError(f"Unsupported language: {language}")

        doc = nlp(text)
        return len(doc)

    def process_document_pair(
        self,
        fr_path: Path,
        de_path: Path,
        celex_id: str,
    ) -> List[SentencePair]:
        """Process a single French-German document pair using numbered paragraph alignment.

        Extracts numbered paragraphs from both French and German documents and aligns
        them by matching paragraph numbers. Documents without numbered paragraphs are
        skipped (this should not occur for CJEU judgments from 2000s onwards).

        Args:
            fr_path: Path to French document
            de_path: Path to German document
            celex_id: CELEX document identifier

        Returns:
            List of aligned paragraph pairs (empty if document lacks numbered paragraphs)

        Raises:
            FileNotFoundError: If document files don't exist
        """
        logger.info(f"Processing {celex_id}...")

        # Read documents
        try:
            with open(fr_path, "r", encoding="utf-8") as f:
                fr_text = f.read()
            with open(de_path, "r", encoding="utf-8") as f:
                de_text = f.read()
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise

        # Extract numbered paragraphs BEFORE cleaning (preserves document structure)
        fr_paragraphs = self.extract_numbered_paragraphs(fr_text)
        de_paragraphs = self.extract_numbered_paragraphs(de_text)

        # Check if both documents have numbered paragraphs
        if fr_paragraphs is None or de_paragraphs is None:
            logger.warning(
                f"{celex_id}: No numbered paragraphs detected, skipping document"
            )
            self.stats["documents_skipped_no_numbering"] += 1
            return []

        # Clean individual paragraphs after extraction
        fr_paragraphs = {num: self.clean_text(text) for num, text in fr_paragraphs.items()}
        de_paragraphs = {num: self.clean_text(text) for num, text in de_paragraphs.items()}

        # Align by numbered paragraphs
        logger.info(
            f"{celex_id}: Detected numbered paragraphs "
            f"(FR: {len(fr_paragraphs)}, DE: {len(de_paragraphs)})"
        )
        aligned_pairs = self.align_numbered_paragraphs(
            fr_paragraphs, de_paragraphs, celex_id
        )

        self.stats["documents_processed"] += 1
        logger.info(f"{celex_id}: Extracted {len(aligned_pairs)} paragraph pairs")

        return aligned_pairs

    def process_all(
        self,
        alignment_index_path: Path,
        output_dir: Path,
        output_filename: str = "parallel_paragraphs.jsonl",
    ) -> None:
        """Process all document pairs and save results.

        Args:
            alignment_index_path: Path to alignment_index.json
            output_dir: Directory to save processed data
            output_filename: Name of output JSONL file

        Raises:
            FileNotFoundError: If alignment index doesn't exist
        """
        logger.info("Starting preprocessing pipeline...")

        # Load alignment index
        try:
            with open(alignment_index_path, "r", encoding="utf-8") as f:
                alignment_index = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Alignment index not found: {e}")
            raise

        logger.info(f"Loaded alignment index with {len(alignment_index)} documents")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        # Process all document pairs
        all_pairs = []

        for celex_id, metadata in alignment_index.items():
            try:
                fr_path = Path(metadata["french_path"])
                de_path = Path(metadata["german_path"])

                pairs = self.process_document_pair(fr_path, de_path, celex_id)
                all_pairs.extend(pairs)

            except Exception as e:
                logger.error(f"Error processing {celex_id}: {e}")
                continue

        # Save to JSON Lines format
        logger.info(f"Saving {len(all_pairs)} paragraph pairs to {output_path}...")

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in all_pairs:
                json.dump(asdict(pair), f, ensure_ascii=False)
                f.write("\n")

        self.stats["final_paragraph_pairs"] = len(all_pairs)

        # Save statistics
        stats_path = output_dir / "preprocessing_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

        logger.info("Preprocessing complete!")
        self._print_statistics()

    def _print_statistics(self) -> None:
        """Print preprocessing statistics to log."""
        logger.info("=" * 60)
        logger.info("PREPROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {self.stats['documents_processed']}")
        logger.info(f"Documents skipped (no numbering): {self.stats['documents_skipped_no_numbering']}")
        logger.info(f"Total paragraph pairs extracted: {self.stats['total_paragraph_pairs']}")
        logger.info(f"Filtered (too short): {self.stats['filtered_too_short']}")
        logger.info(f"Filtered (too long): {self.stats['filtered_too_long']}")
        logger.info(f"Final paragraph pairs: {self.stats['final_paragraph_pairs']}")

        if self.stats['total_paragraph_pairs'] > 0:
            keep_rate = (
                self.stats['final_paragraph_pairs'] /
                self.stats['total_paragraph_pairs'] * 100
            )
            logger.info(f"Keep rate: {keep_rate:.2f}%")

        logger.info("=" * 60)


def main():
    """Main entry point for preprocessing pipeline."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    alignment_index_path = project_root / "data" / "raw" / "metadata" / "alignment_index.json"
    output_dir = project_root / "data" / "processed"

    # Initialize preprocessor
    preprocessor = LegalTextPreprocessor(
        min_tokens=5,
        max_tokens=512,
    )

    # Run preprocessing on modern CJEU judgments (2000s onwards)
    # with numbered paragraph alignment
    preprocessor.process_all(
        alignment_index_path=alignment_index_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
