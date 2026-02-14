# French-German Legal Translation Project
## CJEU Preliminary Reference Rulings Dataset

## Project Overview
Build a neural machine translation system specialized for translating French legal documents 
(specifically CJEU preliminary reference rulings) into German. This is a domain-specific 
translation task requiring high accuracy due to legal implications.

## Dataset Details
- **Source:** CJEU (Court of Justice of the European Union) preliminary reference rulings
- **Access:** Via eurlex R package API at https://michalovadek.github.io/eurlex/
- **Languages:** French (source) → German (target)
- **Domain:** Legal/judicial documents
- **Expected size:** ~10,000-50,000 parallel sentence pairs (to be determined)
- **Quality:** Professional human translations (gold standard)

## Data Characteristics
- Long sentences (legal documents average 30-50 tokens)
- Complex syntactic structures
- Domain-specific legal terminology
- Formal register
- High translation quality (professional translators)

## Technical Stack
- **Framework:** PyTorch + Hugging Face Transformers
- **Model candidates:**
  1. **mBART-large-50** (multilingual, good for French-German, 611M params)
  2. **Helsinki-NLP/opus-mt-fr-de** (baseline, smaller, pre-trained on French-German)
  3. **mT5** (alternative, good for legal text)
- **Data processing:** Hugging Face datasets, custom preprocessing for legal text
- **Evaluation:** BLEU, chrF, TER, plus manual legal terminology accuracy checks
- **Training:** Local development → Google Colab for GPU fine-tuning

## Development Phases

### Phase 1: Data Acquisition (Week 1)
1. Write scraper/API client for eurlex data
2. Download French and German versions of CJEU rulings
3. Align parallel documents (same case ID)
4. Initial data quality assessment

### Phase 2: Data Preprocessing (Week 1-2)
1. Sentence segmentation (legal-aware, handle citations)
2. Alignment verification (ensure French-German pairs match)
3. Cleaning (remove markup, normalize whitespace, handle special characters)
4. Train/validation/test split (stratified by year or case type)
5. Vocabulary analysis (legal term frequency)

### Phase 3: Baseline Model (Week 2-3)
1. Load Helsinki-NLP opus-mt-fr-de as baseline
2. Zero-shot evaluation on test set
3. Error analysis (identify weak areas)

### Phase 4: Fine-tuning (Week 3-4)
1. Prepare mBART-large-50 for fine-tuning
2. Configure training (learning rate, batch size, epochs)
3. Train on Google Colab with GPU
4. Monitor convergence and BLEU scores
5. Checkpoint best models

### Phase 5: Evaluation & Analysis (Week 4-5)
1. Comprehensive evaluation (BLEU, chrF, TER)
2. Legal terminology accuracy analysis
3. Qualitative error analysis (sample 50-100 translations)
4. Compare baseline vs fine-tuned model
5. Document findings

### Phase 6: Deployment & Publication (Week 5-6)
1. Create inference API or script
2. Upload model to Hugging Face Hub
3. Write technical report
4. Publish code and results on GitHub
5. Optional: blog post or paper draft

## Code Style & Conventions
- **Python style:** PEP 8, enforced with Black formatter
- **Type hints:** Use throughout (mypy compatible)
- **Docstrings:** Google style for all functions/classes
- **Testing:** Unit tests for data processing, integration tests for model pipeline
- **Logging:** Use Python logging module, not print statements
- **Config files:** YAML for all hyperparameters and paths

## Repository Structure
```
french-german-legal-translation/
├── data/
│   ├── raw/              # Downloaded CJEU documents
│   ├── processed/        # Cleaned, aligned parallel data
│   └── README.md         # Data documentation and statistics
├── src/
│   ├── data/
│   │   ├── download.py         # Scrape/download from eurlex
│   │   ├── preprocess.py       # Clean and align
│   │   └── dataloader.py       # PyTorch DataLoader
│   ├── models/
│   │   ├── translator.py       # Model wrapper
│   │   └── train.py            # Training script
│   ├── utils/
│   │   ├── helpers.py          # Utility functions
│   │   └── legal_terms.py      # Legal terminology extraction
│   └── evaluate.py             # Evaluation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   └── 03_results_analysis.ipynb
├── configs/
│   ├── model_config.yaml       # Model architecture settings
│   └── training_config.yaml    # Training hyperparameters
├── tests/
│   └── test_preprocessing.py
├── checkpoints/                # Model checkpoints (gitignored)
├── results/                    # Outputs, plots, reports
├── requirements.txt
├── setup.py
├── CLAUDE.md                   # This file
└── README.md
```

## Important Constraints & Guidelines

### Data Handling
- Keep raw data in data/raw/ (separate French and German files)
- Use Git LFS for files >100MB
- Document data provenance (case IDs, dates, URLs)
- Respect CJEU data usage terms (public domain, but cite properly)

### Model Training
- Start with smallest viable model (Helsinki opus-mt) for baseline
- Use gradient accumulation if GPU memory is limited
- Save checkpoints every 1000 steps
- Early stopping based on validation BLEU (patience=3 epochs)
- Log all hyperparameters with Weights & Biases or TensorBoard

### Evaluation
- Primary metric: BLEU (industry standard for translation)
- Secondary: chrF (character-level, robust to morphology)
- Tertiary: TER (translation edit rate)
- **Legal accuracy:** Track translation of key legal terms (create glossary)

### Code Quality
- Run black formatter before commits
- Use type hints for all function signatures
- Write docstrings with examples
- Add unit tests for data processing functions
- No hardcoded paths (use config files or environment variables)

### Security & Privacy
- No sensitive data in git (CJEU data is public, but check for edge cases)
- No API keys in code (use environment variables)
- Docker container should have restricted network access

## Claude Code Interaction Guidelines

### When asking Claude for help:
1. **Be specific about files:** "Update src/data/download.py to add retry logic"
2. **Provide context:** "The eurlex API returns JSON with fields 'celex', 'text', 'language'"
3. **Request tests:** "Also create a test in tests/ that verifies the alignment"
4. **Ask for explanations:** "Explain why mBART is better than mT5 for this task"

### What Claude should prioritize:
- **Correctness over speed** (legal domain requires accuracy)
- **Reproducibility** (seed setting, version pinning)
- **Documentation** (explain legal-specific preprocessing decisions)
- **Modularity** (separate concerns: download, preprocess, train, evaluate)

### What Claude should ask about:
- Ambiguous preprocessing decisions (e.g., how to handle citations)
- Hyperparameter choices (learning rate, batch size trade-offs)
- Evaluation strategy (which metrics matter most for legal domain)
- Error handling (what to do with misaligned or corrupted documents)

## Known Challenges & Considerations

1. **Sentence alignment:** Legal documents have complex structures. Some "sentences" 
   span multiple lines. Consider paragraph-level alignment if sentence-level is too granular.

2. **Legal terminology:** Create a bilingual legal glossary to track term translation accuracy.
   Terms like "Vorabentscheidungsersuchen" (preliminary reference) must be translated correctly.

3. **Citation handling:** Legal documents cite cases, articles, laws. Decide whether to:
   - Translate citations literally
   - Keep citations in original form
   - Special tokenization for citations

4. **Data imbalance:** Some legal topics may be overrepresented. Consider stratified splitting.

5. **Evaluation limitations:** BLEU may not capture legal adequacy. Plan for human evaluation 
   of a sample (50-100 sentences) by a legal expert or bilingual annotator.

## Success Criteria

### Minimum Viable Product (MVP):
- ✅ Successfully download and align 5,000+ parallel sentence pairs
- ✅ Train a baseline model (Helsinki opus-mt) and measure BLEU
- ✅ Fine-tune mBART on CJEU data and achieve >5 BLEU improvement over baseline
- ✅ Document pipeline in README with usage examples

### Stretch Goals:
- 🎯 Achieve BLEU >30 on test set (competitive with general-domain MT)
- 🎯 >90% accuracy on top 100 legal terms
- 🎯 Publish model on Hugging Face Hub with model card
- 🎯 Create interactive demo (Gradio app)
- 🎯 Write technical blog post or paper draft

## Resources & References

### Key Papers:
- "Legal Machine Translation" (Koehn et al.)
- "Domain Adaptation for Neural Machine Translation" (Chu & Wang)
- "mBART: Denoising Sequence-to-Sequence Pre-training for NLG" (Liu et al.)

### Useful Tools:
- eurlex R package: https://michalovadek.github.io/eurlex/
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- SacreBLEU: https://github.com/mjpost/sacrebleu
- Weights & Biases: https://wandb.ai

### Similar Projects:
- OPUS legal corpus: https://opus.nlpl.eu/
- JRC-Acquis (EU legal corpus): https://ec.europa.eu/jrc/en/language-technologies/jrc-acquis

## Current Status
**Phase:** Project initialization
**Next steps:** 
1. Set up development environment (Docker + VS Code)
2. Create data download script for eurlex
3. Explore dataset structure and quality

## Questions for Claude
- How should we handle very long sentences (>512 tokens) in legal documents?
- What's the best approach for handling legal citations in translation?
- Should we use document-level context (multiple sentences) or sentence-level translation?
- How to create a representative train/val/test split for legal documents?

