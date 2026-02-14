"""
Training script for fine-tuning mBART-large-50 on French-German legal translation.

This script fine-tunes the mBART-large-50-many-to-many-mmt model on CJEU legal corpus
for domain-specific translation. It uses Hugging Face Trainer API with mixed precision
training, gradient accumulation, and early stopping.

Usage:
    # Local testing (CPU, small subset)
    python src/models/train.py --debug

    # Full training (GPU recommended)
    python src/models/train.py --output_dir checkpoints/mbart-legal-fr-de

    # Resume from checkpoint
    python src/models/train.py --resume_from_checkpoint checkpoints/mbart-legal-fr-de/checkpoint-500

Example (Google Colab):
    !python src/models/train.py \\
        --output_dir /content/drive/MyDrive/mbart-legal \\
        --num_train_epochs 10 \\
        --per_device_train_batch_size 4
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CustomLoggingCallback(TrainerCallback):
    """Custom callback for enhanced training progress logging.

    Displays:
    - "Starting Epoch X/Y" at the beginning of each epoch
    - "Epoch X/Y - Step A/B - Loss: ..." during training steps
    """

    def __init__(self, num_epochs: int, steps_per_epoch: int):
        """Initialize the callback.

        Args:
            num_epochs: Total number of training epochs
            steps_per_epoch: Number of training steps per epoch
        """
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        self.current_epoch = int(state.epoch) if state.epoch is not None else 0
        logger.info("="*60)
        logger.info(f"Starting Epoch {self.current_epoch + 1}/{self.num_epochs}")
        logger.info("="*60)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs during training."""
        if logs and state.epoch is not None:
            epoch = int(state.epoch) + 1  # 1-indexed for display
            step = state.global_step
            total_steps = self.num_epochs * self.steps_per_epoch

            # Build log message
            log_parts = [f"Epoch {epoch}/{self.num_epochs}"]
            log_parts.append(f"Step {step}/{total_steps}")

            # Add loss if available
            if "loss" in logs:
                log_parts.append(f"Loss: {logs['loss']:.4f}")

            # Add learning rate if available
            if "learning_rate" in logs:
                log_parts.append(f"LR: {logs['learning_rate']:.2e}")

            # Add evaluation metrics if available
            if "eval_loss" in logs:
                log_parts.append(f"Eval Loss: {logs['eval_loss']:.4f}")
            if "eval_bleu" in logs:
                log_parts.append(f"BLEU: {logs['eval_bleu']:.2f}")

            logger.info(" - ".join(log_parts))


@dataclass
class TranslationDataset:
    """Dataset wrapper for French-German parallel data.

    Attributes:
        source_texts: List of French source texts
        target_texts: List of German target texts
        ids: List of paragraph IDs
        celex_ids: List of document IDs
    """
    source_texts: List[str]
    target_texts: List[str]
    ids: List[str]
    celex_ids: List[str]


def load_data_splits(
    data_dir: Path,
    debug: bool = False
) -> Dict[str, TranslationDataset]:
    """Load train/val/test splits from preprocessed data.

    Args:
        data_dir: Path to data/processed directory
        debug: If True, load only small subset for testing

    Returns:
        Dictionary with 'train', 'val', 'test' TranslationDataset objects

    Raises:
        FileNotFoundError: If data files don't exist
    """
    logger.info("Loading data splits...")

    # Load parallel paragraphs
    paragraphs_path = data_dir / "parallel_paragraphs.jsonl"
    if not paragraphs_path.exists():
        raise FileNotFoundError(f"Data file not found: {paragraphs_path}")

    data = []
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} total paragraph pairs")

    # Load split indices
    splits_path = data_dir / "data_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Split indices not found: {splits_path}\n"
            "Please run notebooks/02_baseline_evaluation.ipynb first to create splits."
        )

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Create datasets for each split
    datasets = {}
    for split_name in ["train", "val", "test"]:
        indices = splits[f"{split_name}_indices"]

        # Debug mode: use only first 100 examples
        if debug and split_name == "train":
            indices = indices[:100]
        elif debug:
            indices = indices[:20]

        split_data = [data[i] for i in indices]

        datasets[split_name] = TranslationDataset(
            source_texts=[d["source_text"] for d in split_data],
            target_texts=[d["target_text"] for d in split_data],
            ids=[d["id"] for d in split_data],
            celex_ids=[d["celex_id"] for d in split_data],
        )

        logger.info(f"{split_name.capitalize()}: {len(datasets[split_name].source_texts)} examples")

    return datasets


def prepare_datasets(
    datasets: Dict[str, TranslationDataset],
    tokenizer: MBart50TokenizerFast,
    max_length: int = 512,
    src_lang: str = "fr_XX",
    tgt_lang: str = "de_DE",
) -> DatasetDict:
    """Tokenize and prepare datasets for training.

    Args:
        datasets: Dictionary of TranslationDataset objects
        tokenizer: mBART tokenizer
        max_length: Maximum sequence length
        src_lang: Source language code (mBART format)
        tgt_lang: Target language code (mBART format)

    Returns:
        Hugging Face DatasetDict with tokenized data
    """
    logger.info("Tokenizing datasets...")

    def preprocess_function(examples):
        """Tokenize source and target texts."""
        # Set source language
        tokenizer.src_lang = src_lang

        # Tokenize source texts
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding=False,  # Padding handled by data collator
        )

        # Tokenize target texts
        # Modern approach (transformers >= 4.30): manually switch src_lang for target tokenization
        # The as_target_tokenizer() context manager is deprecated
        original_src_lang = tokenizer.src_lang
        tokenizer.src_lang = tgt_lang
        labels = tokenizer(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        tokenizer.src_lang = original_src_lang  # Restore original source language

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert to Hugging Face Dataset format
    hf_datasets = {}
    for split_name, dataset in datasets.items():
        hf_dataset = Dataset.from_dict({
            "source": dataset.source_texts,
            "target": dataset.target_texts,
            "id": dataset.ids,
            "celex_id": dataset.celex_ids,
        })

        # Tokenize
        hf_dataset = hf_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["source", "target"],
            desc=f"Tokenizing {split_name}",
        )

        hf_datasets[split_name] = hf_dataset

    return DatasetDict(hf_datasets)


def compute_metrics(eval_pred, tokenizer: MBart50TokenizerFast):
    """Compute BLEU and other metrics during evaluation.

    Args:
        eval_pred: EvalPrediction object from Trainer
        tokenizer: Tokenizer for decoding predictions

    Returns:
        Dictionary of metric scores
    """
    # Load metrics
    bleu_metric = evaluate.load("sacrebleu")

    predictions, labels = eval_pred

    # Decode predictions
    # Replace -100 in labels (used for padding) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    try:
        # Attempt to decode predictions and labels
        # Early in training, model may generate invalid token IDs
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]  # BLEU expects list of references

        # Compute BLEU
        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": result["score"],
        }

    except (ValueError, OverflowError, IndexError) as e:
        # Handle decoding errors (e.g., out of range token IDs during early training)
        logger.warning(
            f"Failed to decode predictions during evaluation: {type(e).__name__}: {e}. "
            "This is normal during early training. Returning BLEU=0.0"
        )
        return {
            "bleu": 0.0,
        }


def train(
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
    output_dir: str = "checkpoints/mbart-legal-fr-de",
    data_dir: str = "data/processed",
    src_lang: str = "fr_XX",
    tgt_lang: str = "de_DE",
    max_length: int = 512,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    eval_steps: int = 100,
    save_steps: int = 100,
    logging_steps: int = 50,
    fp16: bool = True,
    early_stopping_patience: int = 3,
    resume_from_checkpoint: Optional[str] = None,
    debug: bool = False,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
) -> None:
    """Fine-tune mBART-large-50 on French-German legal translation.

    Args:
        model_name: Hugging Face model ID
        output_dir: Directory to save model checkpoints
        data_dir: Directory containing preprocessed data
        src_lang: Source language code (mBART format)
        tgt_lang: Target language code (mBART format)
        max_length: Maximum sequence length
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Gradient accumulation steps (effective batch size = batch_size * grad_accum)
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        fp16: Use mixed precision training (faster on GPU)
        early_stopping_patience: Stop if no improvement for N evaluations
        resume_from_checkpoint: Path to checkpoint to resume from
        debug: Use small data subset for quick testing
        push_to_hub: Push model to Hugging Face Hub after training
        hub_model_id: Model ID on Hugging Face Hub (e.g., "username/model-name")
    """
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / data_dir
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("FINE-TUNING MBART FOR LEGAL FRENCH-GERMAN TRANSLATION")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Source language: {src_lang}")
    logger.info(f"Target language: {tgt_lang}")
    logger.info(f"Debug mode: {debug}")
    logger.info("="*60)

    # Load data
    datasets = load_data_splits(data_path, debug=debug)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    # Prepare datasets
    tokenized_datasets = prepare_datasets(
        datasets,
        tokenizer,
        max_length=max_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    # Load model
    logger.info("Loading model...")
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Set generation parameters (modern approach for transformers >= 4.30)
    # decoder_start_token_id stays on config (used during training)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[tgt_lang]
    # forced_bos_token_id goes on generation_config (used during generation/evaluation)
    model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data collator (handles padding dynamically)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Check for existing checkpoint
    last_checkpoint = None
    if resume_from_checkpoint is not None:
        last_checkpoint = resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    elif output_path.exists():
        last_checkpoint = get_last_checkpoint(str(output_path))
        if last_checkpoint is not None:
            logger.info(f"Found existing checkpoint: {last_checkpoint}")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),

        # Training hyperparameters
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,

        # Evaluation and saving
        # Evaluate per epoch but don't save intermediate checkpoints to save disk space
        # Only the final model will be saved at the end (model weights only, ~2.4GB)
        eval_strategy="epoch",
        save_strategy="no",  # No intermediate checkpoints (saves ~6.5GB per checkpoint)
        # Note: load_best_model_at_end disabled since we don't save intermediate checkpoints

        # Logging
        logging_dir=str(output_path / "logs"),
        logging_steps=logging_steps,
        report_to=["tensorboard"],

        # Generation (for evaluation)
        predict_with_generate=True,
        generation_max_length=max_length,
        generation_num_beams=5,

        # Performance optimizations
        fp16=fp16 and torch.cuda.is_available(),
        dataloader_num_workers=4 if not debug else 0,
        dataloader_pin_memory=True,

        # Reproducibility
        seed=42,

        # Hub integration
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    # Calculate training steps for progress logging
    steps_per_epoch = len(tokenized_datasets["train"]) // (
        per_device_train_batch_size * gradient_accumulation_steps
    )
    total_train_steps = steps_per_epoch * num_train_epochs

    logger.info("Training configuration:")
    logger.info(f"  Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    logger.info(f"  Total train steps: ~{total_train_steps}")
    logger.info(f"  Steps per epoch: ~{steps_per_epoch}")
    logger.info(f"  Evaluation: once per epoch ({num_train_epochs} total evaluations)")
    logger.info(f"  Checkpoints: disabled (only final model will be saved)")
    logger.info(f"  Final model size: ~2.4GB (model weights only, no optimizer state)")
    logger.info(f"  FP16: {training_args.fp16}")

    # Initialize custom logging callback
    custom_logger = CustomLoggingCallback(
        num_epochs=num_train_epochs,
        steps_per_epoch=steps_per_epoch
    )

    # Initialize Trainer
    # Note: tokenizer is inferred from data_collator in newer transformers versions
    # Early stopping disabled since we're not saving checkpoints
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=[custom_logger],
    )

    # Train
    logger.info("Starting training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save final model (model weights only, ~2.4GB)
        # Note: trainer.save_model() uses save_pretrained() internally
        # This does NOT save optimizer state, only model weights
        logger.info("Saving final model (weights only, no optimizer state)...")
        trainer.save_model()
        tokenizer.save_pretrained(output_path)

        # Save training metrics (small JSON files)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()  # Saves trainer_state.json (metadata only, ~few KB)

        logger.info("="*60)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Final model saved to: {output_path}")
        logger.info(f"Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info("="*60)

        # Evaluate on test set
        if "test" in tokenized_datasets:
            logger.info("Evaluating on test set...")
            test_results = trainer.predict(tokenized_datasets["test"])
            metrics = test_results.metrics
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

            logger.info("Test set results:")
            logger.info(f"  Loss: {metrics.get('test_loss', 'N/A'):.4f}")
            logger.info(f"  BLEU: {metrics.get('test_bleu', 'N/A'):.2f}")

        # Push to Hub if requested
        if push_to_hub:
            logger.info("Pushing model to Hugging Face Hub...")
            trainer.push_to_hub(commit_message="Fine-tuned mBART on CJEU legal corpus")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info(f"Partial checkpoint saved to: {output_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Fine-tune mBART-large-50 for French-German legal translation"
    )

    # Model and data
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/mbart-legal-fr-de",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="fr_XX",
        help="Source language code (mBART format)",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="de_DE",
        help="Target language code (mBART format)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )

    # Evaluation and checkpointing
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (number of evaluations)",
    )

    # Performance
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable mixed precision training",
    )

    # Checkpointing
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use small data subset for quick testing",
    )

    # Hub integration
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub after training",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID on Hugging Face Hub (e.g., 'username/model-name')",
    )

    args = parser.parse_args()

    # Train
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=not args.no_fp16,
        early_stopping_patience=args.early_stopping_patience,
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug=args.debug,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
