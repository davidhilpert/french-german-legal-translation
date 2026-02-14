"""
Unit tests for preprocessing pipeline.

Tests cover text cleaning, sentence segmentation, alignment,
and filtering logic for legal document preprocessing.
"""

import pytest
from pathlib import Path
from src.data.preprocess import LegalTextPreprocessor, SentencePair


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance for testing."""
    try:
        return LegalTextPreprocessor(min_tokens=5, max_tokens=512)
    except OSError:
        pytest.skip("spaCy models not installed")


class TestTextCleaning:
    """Test text cleaning functionality."""

    def test_clean_text_whitespace(self, preprocessor):
        """Test that excessive whitespace is normalized."""
        text = "Article   123\\n\\n\\nSection  A"
        expected = "Article 123 Section A"
        assert preprocessor.clean_text(text) == expected

    def test_clean_text_unicode(self, preprocessor):
        """Test unicode normalization."""
        text = "Café\\u00A0résumé"  # Non-breaking space
        cleaned = preprocessor.clean_text(text)
        assert "\\u00A0" not in cleaned
        assert "Café" in cleaned

    def test_clean_text_empty(self, preprocessor):
        """Test handling of empty text."""
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text("   \\n\\n  ") == ""

    def test_clean_text_preserves_content(self, preprocessor):
        """Test that meaningful content is preserved."""
        text = "Le système des montants compensatoires."
        cleaned = preprocessor.clean_text(text)
        assert "système" in cleaned
        assert "montants" in cleaned
        assert "compensatoires" in cleaned


class TestSentenceSegmentation:
    """Test sentence segmentation functionality."""

    def test_segment_french(self, preprocessor):
        """Test French sentence segmentation."""
        text = "Première phrase. Deuxième phrase. Troisième phrase."
        sentences = preprocessor.segment_sentences(text, "fr")
        assert len(sentences) == 3
        assert "Première phrase" in sentences[0]

    def test_segment_german(self, preprocessor):
        """Test German sentence segmentation."""
        text = "Erster Satz. Zweiter Satz. Dritter Satz."
        sentences = preprocessor.segment_sentences(text, "de")
        assert len(sentences) == 3
        assert "Erster Satz" in sentences[0]

    def test_segment_invalid_language(self, preprocessor):
        """Test error handling for invalid language."""
        with pytest.raises(ValueError):
            preprocessor.segment_sentences("Text", "en")

    def test_segment_empty_text(self, preprocessor):
        """Test segmentation of empty text."""
        sentences = preprocessor.segment_sentences("", "fr")
        assert len(sentences) == 0


class TestTokenCounting:
    """Test token counting functionality."""

    def test_count_tokens_french(self, preprocessor):
        """Test French token counting."""
        text = "Le système des montants compensatoires."
        count = preprocessor.count_tokens(text, "fr")
        assert count >= 5  # At least 5 tokens

    def test_count_tokens_german(self, preprocessor):
        """Test German token counting."""
        text = "Das System der Ausgleichsbetraege."
        count = preprocessor.count_tokens(text, "de")
        assert count >= 4  # At least 4 tokens

    def test_count_tokens_empty(self, preprocessor):
        """Test token counting on empty text."""
        count = preprocessor.count_tokens("", "fr")
        assert count == 0


class TestSentenceAlignment:
    """Test sentence alignment functionality."""

    def test_align_equal_length(self, preprocessor):
        """Test alignment when sentence counts match."""
        fr_sents = [
            "Première phrase avec assez de mots pour passer le filtre.",
            "Deuxième phrase avec assez de mots pour passer le filtre.",
        ]
        de_sents = [
            "Erster Satz mit genug Woertern um den Filter zu bestehen.",
            "Zweiter Satz mit genug Woertern um den Filter zu bestehen.",
        ]

        pairs = preprocessor.align_sentences(fr_sents, de_sents, "TEST001")

        assert len(pairs) == 2
        assert pairs[0].celex_id == "TEST001"
        assert pairs[0].id == "TEST001_0000"
        assert pairs[1].id == "TEST001_0001"

    def test_align_unequal_length(self, preprocessor):
        """Test alignment when sentence counts differ."""
        fr_sents = [
            "Première phrase avec assez de mots pour passer le filtre.",
            "Deuxième phrase avec assez de mots pour passer le filtre.",
            "Troisième phrase avec assez de mots pour passer le filtre.",
        ]
        de_sents = [
            "Erster Satz mit genug Woertern um den Filter zu bestehen.",
            "Zweiter Satz mit genug Woertern um den Filter zu bestehen.",
        ]

        pairs = preprocessor.align_sentences(fr_sents, de_sents, "TEST002")

        # Should use minimum length
        assert len(pairs) == 2
        # Check that alignment mismatch was tracked
        assert preprocessor.stats["filtered_alignment_mismatch"] > 0

    def test_align_filters_short_sentences(self, preprocessor):
        """Test that too-short sentences are filtered."""
        fr_sents = ["Court.", "Phrase longue avec assez de mots ici."]
        de_sents = ["Kurz.", "Langer Satz mit genug Woertern hier drin."]

        pairs = preprocessor.align_sentences(fr_sents, de_sents, "TEST003")

        # Only the long sentence pair should pass
        assert len(pairs) == 1
        assert preprocessor.stats["filtered_too_short"] > 0

    def test_align_filters_long_sentences(self, preprocessor):
        """Test that too-long sentences are filtered."""
        # Create artificially long sentences
        fr_long = " ".join(["mot"] * 600)  # 600 tokens
        de_long = " ".join(["Wort"] * 600)

        fr_sents = [fr_long, "Phrase normale avec des mots."]
        de_sents = [de_long, "Normaler Satz mit Woertern."]

        pairs = preprocessor.align_sentences(fr_sents, de_sents, "TEST004")

        # Only the normal sentence pair should pass
        assert len(pairs) == 1
        assert preprocessor.stats["filtered_too_long"] > 0


class TestSentencePair:
    """Test SentencePair dataclass."""

    def test_sentence_pair_creation(self):
        """Test creating a SentencePair instance."""
        pair = SentencePair(
            id="TEST_0001",
            source_text="Texte source",
            target_text="Zieltext",
            celex_id="TEST",
            source_tokens=10,
            target_tokens=8,
        )

        assert pair.id == "TEST_0001"
        assert pair.source_text == "Texte source"
        assert pair.target_text == "Zieltext"
        assert pair.celex_id == "TEST"
        assert pair.source_tokens == 10
        assert pair.target_tokens == 8


class TestPreprocessorStatistics:
    """Test statistics tracking."""

    def test_statistics_initialization(self, preprocessor):
        """Test that statistics are initialized correctly."""
        stats = preprocessor.stats
        assert "documents_processed" in stats
        assert "total_sentence_pairs" in stats
        assert "filtered_too_short" in stats
        assert "filtered_too_long" in stats
        assert "filtered_alignment_mismatch" in stats
        assert "final_sentence_pairs" in stats

    def test_statistics_updated_during_alignment(self, preprocessor):
        """Test that statistics are updated during processing."""
        initial_count = preprocessor.stats["total_sentence_pairs"]

        fr_sents = ["Phrase avec assez de mots pour passer."]
        de_sents = ["Satz mit genug Woertern zum Bestehen."]

        preprocessor.align_sentences(fr_sents, de_sents, "TEST005")

        assert preprocessor.stats["total_sentence_pairs"] > initial_count


def test_imports():
    """Test that all required imports work."""
    from src.data.preprocess import (
        LegalTextPreprocessor,
        SentencePair,
        logger,
    )
    assert LegalTextPreprocessor is not None
    assert SentencePair is not None
    assert logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
