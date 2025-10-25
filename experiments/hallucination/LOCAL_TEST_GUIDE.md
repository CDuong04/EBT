# Local Testing Guide

Quick guide to test the hallucination experiment pipeline on your local machine.

## Step 1: Install Additional Dependencies

```bash
# Activate your EBT conda environment
conda activate ebt

# Install analysis dependencies
pip install scikit-learn scipy matplotlib

# Or use the requirements file
pip install -r experiments/hallucination/requirements.txt
```

## Step 2: Run the Test Pipeline

This will verify all components work WITHOUT needing a trained model (uses random initialization):

```bash
cd /oscar/home/cduong5/EBT

python experiments/hallucination/test_pipeline.py
```

### What the test does:

1. ✓ Initializes a tiny EBT model (randomly initialized)
2. ✓ Loads SQuAD validation dataset
3. ✓ Extracts energy values from model forward pass
4. ✓ Generates answers using greedy decoding
5. ✓ Computes F1 and Exact Match scores
6. ✓ Runs full pipeline on 5 samples

### Expected output:

```
================================================================================
TEST 1: Model Initialization
================================================================================
✓ Model initialized successfully
  Parameters: 1,234,567

================================================================================
TEST 2: Dataset Loading
================================================================================
✓ Dataset loaded successfully
  Total samples: 11,873

[... more tests ...]

================================================================================
TEST SUMMARY
================================================================================
✓ ALL TESTS PASSED!
```

## Step 3: Check for Common Issues

### Issue 1: Dataset Download

**Error**: `ConnectionError: Couldn't reach https://huggingface.co/...`

**Solution**:
- Make sure `HF_HOME` environment variable is set
- May need to set `HF_TOKEN` if dataset requires authentication
- Check internet connection

### Issue 2: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Test uses very small model, should work on any GPU
- If still OOM, model size can be reduced in `test_pipeline.py`

### Issue 3: Import Errors

**Error**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
pip install scikit-learn scipy matplotlib
```

## Step 4: Inspect Test Outputs

The test will show you:

1. **Energy extraction example**:
   ```
   Energy trajectory: ['123.45', '98.76', '87.65']
   Initial energy: 123.4500
   Final energy: 87.6500
   Energy gap: 35.8000
   ```

2. **Generation example**:
   ```
   Question: [[Question]]: The Amazon rainforest...
   Generated: random tokens (model is not trained)
   Ground truth: Brazil
   Final energy: 142.3456
   ```

3. **Evaluation example**:
   ```
   Exact Match: 0
   F1 Score: 0.0000
   Correct (F1 >= 0.5): 0
   ```

   Note: Predictions will be wrong since model is randomly initialized!

4. **Pipeline summary**:
   ```
   Sample 1/5: F1=0.00, Energy=142.34
   Sample 2/5: F1=0.15, Energy=138.92
   ...
   Average F1: 0.0500
   Average Energy: 140.2345
   Correct (F1 >= 0.5): 0/5
   ```

## What Success Looks Like

If all tests pass:
```
✓ ALL TESTS PASSED!

You can now proceed to train a real model:
  python experiments/hallucination/train_small_ebt.py
```

## Troubleshooting

### Test fails at model initialization
- Check that base EBT dependencies are installed
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### Test fails at dataset loading
- Check HF_HOME is set: `echo $HF_HOME`
- Try downloading manually: `python -c "from datasets import load_dataset; load_dataset('rajpurkar/squad_v2')"`

### Test succeeds but predictions are all wrong
- **This is expected!** Model is randomly initialized
- After training, predictions should improve

## Next Steps

Once tests pass:

1. **Train a model**: `python experiments/hallucination/train_small_ebt.py`
   - Takes 2-4 hours on 1 GPU
   - Saves checkpoint to `experiments/hallucination/checkpoints/`

2. **Run full experiment**: See `README.md` for complete pipeline

## Quick Sanity Checks

Before starting the full experiment, verify:

- [ ] All 6 tests pass
- [ ] No CUDA errors
- [ ] Dataset loads successfully
- [ ] Energy values are extracted (non-zero numbers)
- [ ] F1/EM scores are computed
- [ ] No import errors

If all checks pass → Ready to train!
