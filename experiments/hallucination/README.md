# Energy-as-Uncertainty for Hallucination Detection

This experiment tests whether EBT's energy values correlate with factual correctness on SQuAD question answering.

## Quick Start: Testing Locally

Before running the full experiment, test on a tiny subset:

```bash
cd /oscar/home/cduong5/EBT

# 1. Quick test with dummy checkpoint (to verify code works)
python experiments/hallucination/test_pipeline.py
```

This will verify all components work without needing a trained model.

## Full Experiment Pipeline

### Phase 1: Train Small EBT Model

Train a small EBT on SQuAD (takes ~2-4 hours on 1 GPU):

```bash
python experiments/hallucination/train_small_ebt.py
```

**Output**: Checkpoint saved to `experiments/hallucination/checkpoints/`

**Hyperparameters**:
- Model size: 4 layers, 256 dim (very small for fast training)
- Training steps: 5,000
- Dataset: SQuAD v2 (finetune mode)
- Validation every 500 steps

### Phase 2: Generate Predictions with Energy Tracking

Generate answers on validation set and track energy values:

```bash
python experiments/hallucination/generate_with_energy.py \
  --checkpoint_path experiments/hallucination/checkpoints/ebt_squad_XXXX.ckpt \
  --output_file experiments/hallucination/results/predictions_with_energy.jsonl \
  --num_samples 1000 \
  --max_gen_len 50
```

**Output**: JSONL file with predictions and energy metrics

Each line contains:
```json
{
  "idx": 0,
  "question": "[[Question]]: context... question?\n[[Answer]]: ",
  "generated_answer": "the generated answer",
  "ground_truth": "correct answer",
  "energy_final": 123.45,
  "energy_initial": 234.56,
  "energy_gap": 111.11,
  "energy_max": 250.00,
  "all_energies": [[...], [...]]
}
```

### Phase 3: Evaluate Predictions

Compute F1 and Exact Match scores, add correctness labels:

```bash
python experiments/hallucination/evaluate_predictions.py \
  --input_file experiments/hallucination/results/predictions_with_energy.jsonl \
  --output_file experiments/hallucination/results/predictions_labeled.jsonl \
  --f1_threshold 0.5
```

**Output**: JSONL with added fields:
- `exact_match`: Binary (1 if perfect match)
- `f1`: F1 score (0-1)
- `correct`: Binary (1 if F1 >= 0.5)

**Prints**:
- Average Exact Match %
- Average F1 Score %
- Accuracy (F1 >= 0.5) %
- Sample correct/incorrect examples

### Phase 4: Analyze Energy-Hallucination Correlation

Compute AUROC and create visualizations:

```bash
python experiments/hallucination/analyze_energy_correlation.py \
  --input_file experiments/hallucination/results/predictions_labeled.jsonl \
  --output_dir experiments/hallucination/results/analysis
```

**Output Files**:
1. `roc_curves.png` - ROC curves for each energy metric
2. `energy_distributions.png` - Histograms of energy for correct vs incorrect
3. `precision_recall_curves.png` - PR curves for rejection strategies
4. `energy_vs_f1_scatter.png` - Scatter plot of energy vs F1
5. `analysis_report.txt` - Summary report with AUROC scores

**Prints**:
- AUROC for each energy metric (Final, Initial, Gap, Max)
- Mean energy for correct vs incorrect predictions
- Rejection analysis (what % of errors caught at different thresholds)

## Energy Metrics Tested

1. **Final Energy**: Energy after all MCMC steps (primary metric)
2. **Initial Energy**: Energy before MCMC refinement
3. **Energy Gap**: Initial - Final (how much refinement occurred)
4. **Max Energy**: Maximum energy across all MCMC steps

## Success Criteria

**Phase 1 is successful if:**
- ✓ AUROC > 0.6 (random = 0.5)
- ✓ EBT energy outperforms baseline confidence
- ✓ Visual separation in energy distributions
- ✓ Actionable rejection threshold exists

**If successful** → Proceed to Phase 2 (intervention experiments)

**If not successful** → Try:
- Different energy aggregation (per-token max instead of mean)
- More MCMC steps during inference
- Contrastive fine-tuning with correct/incorrect pairs

## File Structure

```
experiments/hallucination/
├── README.md                          # This file
├── train_small_ebt.py                 # Phase 1: Training
├── generate_with_energy.py            # Phase 2: Generation + energy tracking
├── evaluate_predictions.py            # Phase 3: Compute F1/EM scores
├── analyze_energy_correlation.py      # Phase 4: AUROC analysis
├── test_pipeline.py                   # Quick local test
├── checkpoints/                       # Saved model checkpoints
│   └── ebt_squad_XXXX.ckpt
└── results/                           # Experiment outputs
    ├── predictions_with_energy.jsonl  # Raw predictions + energies
    ├── predictions_labeled.jsonl      # With correctness labels
    └── analysis/                      # Plots and reports
        ├── roc_curves.png
        ├── energy_distributions.png
        ├── precision_recall_curves.png
        ├── energy_vs_f1_scatter.png
        └── analysis_report.txt
```

## Expected Runtime

| Phase | Time (1 GPU) | Output Size |
|-------|-------------|-------------|
| Training | 2-4 hours | ~500 MB checkpoint |
| Generation (1000 samples) | 30-60 min | ~5 MB JSONL |
| Evaluation | < 1 min | ~5 MB JSONL |
| Analysis | < 1 min | ~5 MB plots |

## Troubleshooting

**Issue**: Model not converging during training
- Solution: Check wandb logs, may need to adjust learning rate or train longer

**Issue**: OOM during generation
- Solution: Reduce batch size to 1, reduce max_gen_len

**Issue**: All predictions are incorrect
- Solution: Model may be undertrained, check validation loss during training

**Issue**: Energy values are all very similar
- Solution: May need more MCMC steps or different MCMC hyperparameters

## Next Steps (Phase 2)

If Phase 1 shows energy correlates with errors:

1. **Rejection Threshold**: Refuse to answer when energy > threshold
2. **Adaptive MCMC**: Run more refinement steps for high-energy predictions
3. **Multi-Sample Voting**: Generate multiple answers, select lowest energy
4. **Contrastive Training**: Explicitly train energy to separate correct/incorrect
