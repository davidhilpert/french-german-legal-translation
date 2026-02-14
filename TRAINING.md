# Training Guide: Fine-tuning mBART for Legal Translation

This guide explains how to fine-tune mBART-large-50 on the CJEU French-German legal corpus.

## Prerequisites

### 1. Complete Data Preprocessing

First, ensure you have preprocessed data and created train/val/test splits:

```bash
# Run preprocessing
python -m src.data.preprocess

# Create splits (run baseline evaluation notebook)
jupyter notebook notebooks/02_baseline_evaluation.ipynb
```

This should create:
- `data/processed/parallel_paragraphs.jsonl` (1,055 paragraph pairs)
- `data/processed/data_splits.json` (train/val/test indices)

### 2. Install Dependencies

```bash
pip install transformers[torch] datasets evaluate sacrebleu sentencepiece protobuf accelerate tensorboard
```

## Training Options

### Option 1: Local CPU (Debug Mode Only)

**Use for:** Quick testing and debugging
**Time:** ~5 minutes for small subset
**Not recommended for:** Full training (too slow)

```bash
python src/models/train.py \
    --debug \
    --output_dir checkpoints/mbart-legal-test \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_steps 20 \
    --no_fp16
```

### Option 2: Local GPU

**Use for:** If you have a GPU with 16GB+ VRAM
**Time:** ~1-2 hours
**Recommended for:** Final training runs

```bash
python src/models/train.py \
    --output_dir checkpoints/mbart-legal-fr-de \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 50 \
    --early_stopping_patience 3
```

### Option 3: Google Colab (Free GPU)

**Use for:** Most users (best option!)
**Time:** ~2-4 hours on T4 GPU
**Recommended for:** Full training

1. Open `notebooks/03_train_mbart_colab.ipynb` in Google Colab
2. Set Runtime → Change runtime type → GPU (T4)
3. Upload your `data/processed/` folder to Google Drive
4. Follow the notebook instructions

## Training Configuration Explained

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 10 | Total training epochs |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch = 4×4 = 16 |
| `learning_rate` | 5e-5 | AdamW learning rate |
| `warmup_steps` | 500 | Linear warmup steps |
| `early_stopping_patience` | 3 | Stop if no improvement for 3 evals |
| `eval_steps` | 100 | Evaluate every 100 steps |
| `save_steps` | 100 | Save checkpoint every 100 steps |
| `max_length` | 512 | Max sequence length (tokens) |

### Memory Requirements

**Model size:** mBART-large-50 = 611M parameters (~2.4GB)

**Training memory (per batch):**
- Batch size 1: ~6GB VRAM
- Batch size 4: ~12GB VRAM (recommended)
- Batch size 8: ~20GB VRAM (requires A100)

**If you get OOM (Out of Memory):**
```bash
# Reduce batch size
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8  # Keep effective batch size = 16
```

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir checkpoints/mbart-legal-fr-de/logs
```

Navigate to `http://localhost:6006` to see:
- Training loss curve
- Validation loss curve
- Learning rate schedule
- BLEU scores during evaluation

### Training Logs

Watch logs in terminal:

```bash
tail -f checkpoints/mbart-legal-fr-de/logs/events.out.tfevents.*
```

## Resuming Training

If training is interrupted (e.g., Colab session timeout), resume from the last checkpoint:

```bash
python src/models/train.py \
    --output_dir checkpoints/mbart-legal-fr-de \
    --resume_from_checkpoint checkpoints/mbart-legal-fr-de/checkpoint-500 \
    --num_train_epochs 10
```

The script automatically detects the last checkpoint if you don't specify one.

## Expected Results

### Training Progress

Typical training curves:
- **Epoch 1-2:** Train loss drops rapidly (6.0 → 2.5)
- **Epoch 3-5:** Loss plateaus (2.5 → 1.5)
- **Epoch 6-10:** Gradual improvement (1.5 → 1.0)

### Target Metrics (on validation set)

| Metric | Baseline | Target After Fine-tuning |
|--------|----------|--------------------------|
| BLEU | ~15-25 | **30-40** (+10-15 points) |
| chrF | ~40-50 | **55-65** |
| Legal term accuracy | ~60-75% | **>90%** |

### Early Stopping

Training stops automatically if validation loss doesn't improve for 3 consecutive evaluations (~300 steps).

## After Training

### 1. Test the Model

```python
from src.models.translator import Translator

translator = Translator("checkpoints/mbart-legal-fr-de")
translation = translator.translate("Votre texte français ici...")
print(translation)
```

### 2. Evaluate on Test Set

Run evaluation notebook:
```bash
jupyter notebook notebooks/02_baseline_evaluation.ipynb
```

Modify the notebook to load your fine-tuned model instead of the baseline.

### 3. Share on Hugging Face Hub (Optional)

```bash
python src/models/train.py \
    --output_dir checkpoints/mbart-legal-fr-de \
    --push_to_hub \
    --hub_model_id YOUR_USERNAME/mbart-legal-fr-de
```

## Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size:
   ```bash
   --per_device_train_batch_size 2
   ```
2. Reduce sequence length:
   ```bash
   --max_length 256
   ```
3. Enable gradient checkpointing (slower but saves memory):
   ```bash
   --gradient_checkpointing True
   ```

### Training Too Slow

**On CPU:** Use Google Colab GPU instead
**On GPU:** Check GPU utilization with `nvidia-smi`. If low (<80%), increase batch size.

### Poor Results After Training

**Possible causes:**
1. **Learning rate too high:** Try `--learning_rate 2e-5`
2. **Not enough epochs:** Increase to `--num_train_epochs 15`
3. **Data issues:** Check data quality in exploration notebook
4. **Overfitting:** Reduce epochs or add weight decay `--weight_decay 0.01`

### Data Splits Not Found

**Error:** `FileNotFoundError: data/processed/data_splits.json`

**Solution:** Run baseline evaluation notebook first to create splits:
```bash
jupyter notebook notebooks/02_baseline_evaluation.ipynb
```

## Advanced Usage

### Custom Hyperparameters

```bash
python src/models/train.py \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --num_train_epochs 15 \
    --early_stopping_patience 5 \
    --gradient_accumulation_steps 8
```

### Different mBART Variant

```bash
# Use smaller model (faster, less accurate)
python src/models/train.py \
    --model_name facebook/mbart-large-50
```

### Save to Google Drive (Colab)

```bash
python src/models/train.py \
    --output_dir /content/drive/MyDrive/mbart-legal-fr-de
```

## Compute Requirements Summary

| Setup | Hardware | Time | Cost |
|-------|----------|------|------|
| Local CPU | Any | ~20+ hours | Free |
| Local GPU (RTX 3080) | 16GB VRAM | ~1-2 hours | Free |
| Google Colab Free | T4 GPU (15GB) | ~2-4 hours | Free |
| Google Colab Pro | V100 GPU (32GB) | ~1-2 hours | $10/month |
| Kaggle GPU | P100 (16GB) | ~2-3 hours | Free |

**Recommendation:** Google Colab Free GPU is the best option for most users.

## Next Steps

After training completes:
1. ✅ Evaluate on test set (compare with baseline)
2. ✅ Test on new legal documents
3. ✅ Create a demo (Gradio/Streamlit)
4. ✅ Write technical report
5. ✅ Share model on Hugging Face Hub

Good luck with training! 🚀
