# Phase 1 Implementation Summary

## What Was Implemented

Complete pipeline for testing whether EBT energy correlates with hallucinations on SQuAD question answering.

### Files Created

```
experiments/hallucination/
├── README.md                          # Complete experiment documentation
├── LOCAL_TEST_GUIDE.md                # Quick testing guide
├── IMPLEMENTATION_SUMMARY.md          # This file
├── requirements.txt                   # Additional dependencies
│
├── train_small_ebt.py                 # Phase 1: Train small EBT on SQuAD
├── generate_with_energy.py            # Phase 2: Generate + track energies
├── evaluate_predictions.py            # Phase 3: Compute F1/EM scores
├── analyze_energy_correlation.py      # Phase 4: AUROC analysis
└── test_pipeline.py                   # Local testing script
```

## Component Details

### 1. Training Script (`train_small_ebt.py`)

**Purpose**: Train a small EBT model on SQuAD for fast experimentation

**Features**:
- Tiny model: 4 layers, 256 dim, 4 heads (~1M parameters)
- Fine-tuning mode on SQuAD v2
- 5,000 training steps (~2-4 hours on 1 GPU)
- Saves checkpoints to `experiments/hallucination/checkpoints/`
- Logs to Weights & Biases

**Key hyperparameters**:
- `mcmc_num_steps=2`: Two MCMC refinement steps
- `mcmc_step_size=500.0`: Initial step size for energy minimization
- `ebt_type="time_embed"`: Conditions on MCMC step

**Usage**:
```bash
python experiments/hallucination/train_small_ebt.py
```

### 2. Generation Script (`generate_with_energy.py`)

**Purpose**: Generate answers on validation set and extract energy metrics

**Features**:
- Greedy decoding (deterministic for reproducibility)
- Tracks 4 energy metrics:
  - `energy_final`: Energy after all MCMC steps
  - `energy_initial`: Energy before MCMC
  - `energy_gap`: Initial - Final (refinement amount)
  - `energy_max`: Maximum across MCMC steps
- Saves results as JSONL (one sample per line)
- Progress bar for long runs

**Key functions**:
- `extract_energies_from_forward()`: Extracts energy from model output
- `generate_answer_greedy()`: Autoregressive generation with energy tracking

**Usage**:
```bash
python experiments/hallucination/generate_with_energy.py \
  --checkpoint_path experiments/hallucination/checkpoints/ebt_squad_XXXX.ckpt \
  --num_samples 1000 \
  --max_gen_len 50
```

**Output format** (JSONL):
```json
{
  "idx": 0,
  "question": "[[Question]]: context... question text\n[[Answer]]: ",
  "generated_answer": "the generated answer",
  "ground_truth": "correct answer",
  "energy_final": 123.45,
  "energy_initial": 234.56,
  "energy_gap": 111.11,
  "energy_max": 250.00,
  "all_energies": [[...], [...]]
}
```

### 3. Evaluation Script (`evaluate_predictions.py`)

**Purpose**: Compute correctness metrics and add labels

**Features**:
- **Exact Match (EM)**: Binary score (1 if perfect match after normalization)
- **F1 Score**: Token overlap metric (0-1)
- **Binary Label**: `correct = 1` if F1 >= threshold (default 0.5)
- Normalization: lowercases, removes punctuation/articles
- Shows sample correct/incorrect predictions

**Key functions**:
- `normalize_answer()`: Standardizes answer format
- `compute_exact_match()`: Binary correctness
- `compute_f1()`: Token overlap F1

**Usage**:
```bash
python experiments/hallucination/evaluate_predictions.py \
  --input_file experiments/hallucination/results/predictions_with_energy.jsonl \
  --f1_threshold 0.5
```

**Output additions**:
- `exact_match`: 0 or 1
- `f1`: 0.0 to 1.0
- `correct`: 0 or 1 (based on F1 threshold)

### 4. Analysis Script (`analyze_energy_correlation.py`)

**Purpose**: Compute AUROC and create visualizations

**Features**:
- **AUROC computation**: For all 4 energy metrics
- **Statistical tests**: t-tests, Pearson correlation
- **Visualizations**:
  - ROC curves (4 metrics)
  - Energy distributions (correct vs incorrect)
  - Precision-Recall curves
  - Energy vs F1 scatter plot
- **Rejection analysis**: What % errors caught at different thresholds
- **Summary report**: Text file with interpretation

**Key functions**:
- `compute_auroc_for_metrics()`: Calculate AUROC for each energy type
- `plot_energy_distributions()`: Histograms with t-test p-values
- `analyze_rejection_threshold()`: Rejection strategy analysis
- `generate_report()`: Auto-interpretation of results

**Usage**:
```bash
python experiments/hallucination/analyze_energy_correlation.py \
  --input_file experiments/hallucination/results/predictions_labeled.jsonl
```

**Output files**:
1. `roc_curves.png`: 2x2 grid of ROC curves
2. `energy_distributions.png`: Overlapping histograms
3. `precision_recall_curves.png`: PR curves for rejection
4. `energy_vs_f1_scatter.png`: Scatter with correlation
5. `analysis_report.txt`: Summary with recommendations

### 5. Test Pipeline (`test_pipeline.py`)

**Purpose**: Verify all components work without trained model

**Features**:
- Uses randomly initialized model (no training needed)
- Tests all 6 components sequentially
- Runs on 5 samples for speed
- Prints detailed error messages if anything fails
- Safe to run repeatedly

**Tests**:
1. Model initialization (checks GPU, parameters)
2. Dataset loading (SQuAD v2 validation)
3. Energy extraction (forward pass)
4. Generation pipeline (greedy decoding)
5. Evaluation metrics (F1/EM)
6. Full pipeline (all components together)

**Usage**:
```bash
python experiments/hallucination/test_pipeline.py
```

## Success Criteria

### Phase 1 is Successful If:

1. **AUROC > 0.6** (random = 0.5)
   - Indicates energy has predictive power for errors

2. **Clear separation in distributions**
   - Mean energy for incorrect > mean energy for correct
   - Statistically significant (p < 0.05)

3. **Actionable rejection threshold**
   - Example: Reject top 20% energy → catch 60%+ errors at 80%+ precision

4. **Visual correlation**
   - Scatter plot shows negative correlation: higher energy → lower F1

### If Successful → Next Steps:

1. **Phase 2A**: Implement rejection threshold
2. **Phase 2B**: Adaptive MCMC (more steps for high energy)
3. **Phase 2C**: Multi-sample voting (select lowest energy)

### If Not Successful → Debugging:

1. Try different energy aggregation:
   - Per-token max instead of mean
   - Energy variance across tokens
   - Energy from specific MCMC steps

2. Increase MCMC steps:
   - Change `mcmc_num_steps=2` to `4` or `8`
   - More refinement may improve signal

3. Contrastive fine-tuning:
   - Create dataset of correct/incorrect pairs
   - Train energy to separate them explicitly

## Dataset Choice: SQuAD v2

**Why SQuAD?**
- ✓ Clear ground truth (extractive answers)
- ✓ Both binary (EM) and continuous (F1) metrics
- ✓ Already has dataloader in codebase
- ✓ 11,873 validation samples (good for statistics)
- ✓ No manual evaluation needed

**Format**:
- Input: Context paragraph + question
- Output: Short answer span
- Example:
  - Q: "Who founded Apple?"
  - GT: "Steve Jobs"
  - Generated: "Steve Jobs and Steve Wozniak"
  - EM: 0, F1: 0.67, Correct: 1

## Energy Metrics Explained

1. **Final Energy**
   - Energy after all MCMC refinement steps
   - **Hypothesis**: Higher = model uncertain about answer

2. **Initial Energy**
   - Energy of noisy initial condition
   - Usually high, less informative

3. **Energy Gap**
   - Initial - Final
   - **Hypothesis**: Larger gap = more refinement needed = harder question

4. **Max Energy**
   - Maximum across all MCMC steps
   - **Hypothesis**: Peak energy indicates difficulty

**Expected Best Metric**: Final Energy or Energy Gap

## Expected Runtime

| Task | Time (1 GPU) | Output |
|------|-------------|--------|
| Test pipeline | 2-5 min | Terminal output |
| Training | 2-4 hours | ~500 MB checkpoint |
| Generation (1000 samples) | 30-60 min | ~5 MB JSONL |
| Evaluation | < 1 min | ~5 MB JSONL |
| Analysis | < 1 min | Plots + report |

**Total**: ~3-5 hours for complete experiment

## How to Run Complete Experiment

### Step 0: Test Locally (2-5 min)
```bash
python experiments/hallucination/test_pipeline.py
```

### Step 1: Train Model (2-4 hours)
```bash
python experiments/hallucination/train_small_ebt.py
```
Output: `experiments/hallucination/checkpoints/last.ckpt`

### Step 2: Generate Predictions (30-60 min)
```bash
python experiments/hallucination/generate_with_energy.py \
  --checkpoint_path experiments/hallucination/checkpoints/last.ckpt \
  --num_samples 1000
```
Output: `experiments/hallucination/results/predictions_with_energy.jsonl`

### Step 3: Evaluate (< 1 min)
```bash
python experiments/hallucination/evaluate_predictions.py
```
Output: `experiments/hallucination/results/predictions_labeled.jsonl`

### Step 4: Analyze (< 1 min)
```bash
python experiments/hallucination/analyze_energy_correlation.py
```
Output: `experiments/hallucination/results/analysis/` (plots + report)

### Step 5: Check Results

Open `experiments/hallucination/results/analysis/analysis_report.txt`

Look for:
- AUROC scores (want > 0.6)
- Recommendation (proceed to Phase 2 or pivot)

## Key Design Decisions

1. **Small model**: Fast iteration > high accuracy
2. **SQuAD v2**: Objective evaluation, no human labeling
3. **Greedy decoding**: Deterministic, reproducible
4. **F1 >= 0.5**: Lenient threshold (counts partial credit)
5. **1000 samples**: Balance speed vs statistics
6. **4 energy metrics**: Test multiple hypotheses

## Modifications to Original EBT Code

**None!** All code is new, no modifications to core EBT implementation needed.

The experiment uses EBT as-is and only adds:
- Wrapper for training on SQuAD
- Scripts to extract and analyze energies
- Evaluation and analysis tools

This is intentional for minimal experiment design.

## Common Issues & Solutions

### Issue 1: Training doesn't converge
**Symptom**: Validation loss doesn't decrease
**Solution**: Check wandb logs, may need longer training or higher LR

### Issue 2: All predictions are empty
**Symptom**: Generated answers are blank
**Solution**: Model may be undertrained, check validation perplexity

### Issue 3: Energy values are all similar
**Symptom**: AUROC ≈ 0.5, no separation in distributions
**Solution**:
- Increase MCMC steps
- Try different energy metric (gap instead of final)
- May need contrastive training

### Issue 4: OOM during generation
**Symptom**: CUDA out of memory
**Solution**: Reduce max_gen_len, use batch_size=1

## What's Next?

After Phase 1 completes:

1. **Check AUROC** in analysis_report.txt
2. **If AUROC > 0.6**: Design Phase 2 intervention
3. **If AUROC < 0.6**: Debug energy extraction or try contrastive training

## Questions?

See:
- `README.md` for full documentation
- `LOCAL_TEST_GUIDE.md` for testing help
- Issues in code? Check `test_pipeline.py` output
