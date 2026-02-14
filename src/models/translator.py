"""
Translation model wrapper for neural machine translation.

This module provides a unified interface for loading and using Hugging Face
translation models (Helsinki-NLP, mBART, etc.) with support for single and
batch translation, beam search, and progress tracking.

Example:
    >>> from src.models.translator import Translator
    >>> translator = Translator("Helsinki-NLP/opus-mt-fr-de")
    >>> translation = translator.translate("Bonjour le monde!")
    >>> print(translation)
    'Hallo Welt!'
"""

import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Translator:
    """Wrapper class for neural machine translation models.

    Provides a unified interface for Hugging Face seq2seq models with support
    for single and batch translation, beam search configuration, and GPU acceleration.

    Attributes:
        model_name: Name or path of the Hugging Face model
        model: Loaded seq2seq model
        tokenizer: Loaded tokenizer
        device: Device for inference (cuda/cpu)
        max_length: Maximum sequence length for generation
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the translator with a pre-trained model.

        Args:
            model_name: Hugging Face model ID (e.g., "Helsinki-NLP/opus-mt-fr-de")
                       or local path to a fine-tuned model
            device: Device to use for inference ("cuda" or "cpu"). If None, automatically
                   detects GPU availability
            max_length: Maximum sequence length for generation (default: 512)
            cache_dir: Directory to cache downloaded models (optional)

        Raises:
            ValueError: If model_name is invalid or model cannot be loaded
        """
        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")

        try:
            # Load tokenizer
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )

            # Load model
            self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info(f"Model loaded successfully: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ValueError(f"Could not load model {model_name}") from e

    def translate(
        self,
        text: str,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs,
    ) -> str:
        """Translate a single text string.

        Args:
            text: Source text to translate
            num_beams: Number of beams for beam search (default: 5)
            length_penalty: Exponential penalty to sequence length (default: 1.0).
                           Values > 1.0 favor longer sequences, < 1.0 favor shorter
            early_stopping: Whether to stop generation when all beams finish
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Translated text as a string

        Example:
            >>> translator = Translator("Helsinki-NLP/opus-mt-fr-de")
            >>> translation = translator.translate("Bonjour!")
            >>> print(translation)
            'Hallo!'
        """
        if not text or not text.strip():
            logger.warning("Empty input text, returning empty string")
            return ""

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                **kwargs,
            )

        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translation

    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> List[str]:
        """Translate a batch of texts with progress tracking.

        Processes texts in batches for efficiency. Empty or whitespace-only
        texts are returned as empty strings.

        Args:
            texts: List of source texts to translate
            batch_size: Number of texts to process in parallel (default: 8)
            num_beams: Number of beams for beam search (default: 5)
            length_penalty: Exponential penalty to sequence length (default: 1.0)
            early_stopping: Whether to stop generation when all beams finish
            show_progress: Whether to display progress bar (default: True)
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            List of translated texts (same length as input)

        Example:
            >>> translator = Translator("Helsinki-NLP/opus-mt-fr-de")
            >>> sources = ["Bonjour!", "Comment allez-vous?"]
            >>> translations = translator.translate_batch(sources)
            >>> print(translations)
            ['Hallo!', 'Wie geht es Ihnen?']
        """
        if not texts:
            logger.warning("Empty text list provided")
            return []

        translations = []

        # Create progress bar
        progress_bar = tqdm(
            total=len(texts),
            desc="Translating",
            disable=not show_progress,
        )

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Filter out empty texts but keep track of their positions
            non_empty_indices = [j for j, text in enumerate(batch) if text.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]

            if non_empty_texts:
                # Tokenize batch
                inputs = self.tokenizer(
                    non_empty_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                # Generate translations
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        early_stopping=early_stopping,
                        **kwargs,
                    )

                # Decode outputs
                batch_translations = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]

                # Reconstruct batch with empty strings for empty inputs
                full_batch_translations = []
                trans_idx = 0
                for j in range(len(batch)):
                    if j in non_empty_indices:
                        full_batch_translations.append(batch_translations[trans_idx])
                        trans_idx += 1
                    else:
                        full_batch_translations.append("")

                translations.extend(full_batch_translations)
            else:
                # All texts in batch are empty
                translations.extend([""] * len(batch))

            progress_bar.update(len(batch))

        progress_bar.close()

        return translations

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model metadata:
                - model_name: Name or path of the model
                - device: Device used for inference
                - max_length: Maximum sequence length
                - num_parameters: Total number of model parameters
                - num_trainable_parameters: Number of trainable parameters
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "num_parameters": total_params,
            "num_trainable_parameters": trainable_params,
        }

    def __repr__(self) -> str:
        """String representation of the Translator."""
        return (
            f"Translator(model_name='{self.model_name}', "
            f"device='{self.device}', max_length={self.max_length})"
        )


def load_translator(
    model_name: str,
    device: Optional[str] = None,
    **kwargs,
) -> Translator:
    """Convenience function to load a translator.

    Args:
        model_name: Hugging Face model ID or local path
        device: Device to use for inference (optional)
        **kwargs: Additional arguments passed to Translator()

    Returns:
        Initialized Translator instance

    Example:
        >>> translator = load_translator("Helsinki-NLP/opus-mt-fr-de")
        >>> translation = translator.translate("Bonjour!")
    """
    return Translator(model_name=model_name, device=device, **kwargs)
