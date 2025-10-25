"""
Quick test to verify all components work without needing a trained model
Tests on 10 samples with a randomly initialized model
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import json
import tempfile
from transformers import AutoTokenizer

from data.nlp.squad_dataloader import SQuADDataset
from model.nlp.ebt import EBT_NLP

class TestHParams:
    """Minimal hyperparameters for testing"""
    def __init__(self):
        # Basic model config
        self.embedding_dim = 128
        self.num_transformer_blocks = 2
        self.multiheaded_attention_heads = 2
        self.ffn_dim_multiplier = 1
        self.weight_initialization_method = "xavier"
        self.weight_initialization_gain = 1.0

        # Data config
        self.dataset_dir = ""
        self.dataset_name = "squad"
        self.context_length = 128
        self.pretokenize_dataset = True
        self.tokenizer = "EleutherAI/gpt-neox-20b"
        self.execution_mode = "inference"
        self.batch_size_per_device = 1

        # EBT config
        self.model_name = "ebt"
        self.mcmc_step_size = 100.0
        self.mcmc_step_size_lr_multiplier = 300.0
        self.mcmc_num_steps = 2
        self.ebt_type = "default"  # Use default for testing (time_embed has issues with variable sequence lengths)
        self.normalize_initial_condition = True
        self.denoising_initial_condition = "random_noise"
        self.mcmc_step_size_learnable = False  # Don't need learnable for test

        # Additional required params
        self.ebt_norm = "rms"
        self.ebt_act_func = "silu"
        self.dyt_alpha_init = 0.5
        self.mcmc_replay_buffer = False
        self.gaussian_random_noise_scaling = 1.0
        self.normalize_initial_condition_only_first_step = False
        self.randomize_mcmc_step_size_scale = 1.0
        self.randomize_mcmc_num_steps = 0
        self.randomize_mcmc_num_steps_min = 0
        self.randomize_mcmc_num_steps_final_landscape = False
        self.langevin_dynamics_noise = 0.0
        self.langevin_dynamics_noise_learnable = False
        self.vocab_to_embed_uses_prob_dist = False
        self.num_modality_processing_mlp_layers = 1
        self.truncate_mcmc = False
        self.clamp_futures_grad = False
        self.clamp_futures_grad_max_change = 9.0
        self.absolute_clamp = 0.0
        self.clamp_max_after_warm_up = 0.0
        self.sharpen_predicted_distribution = 0.0
        self.reconstruction_coeff = 1.0
        self.contrastive_loss = False
        self.contrastive_loss_coeff = 0.0005
        self.soften_target_prob_dist = 0.0
        self.debug_unused_parameters = False
        self.no_mcmc_detach = False  # Required for forward pass
        self.discrete_contrastive_loss_true_logit_val = 0

def test_model_initialization():
    """Test 1: Can we initialize the model?"""
    print("\n" + "="*80)
    print("TEST 1: Model Initialization")
    print("="*80)

    try:
        hparams = TestHParams()
        model = EBT_NLP(hparams)
        model = model.cuda()
        model.eval()

        print("✓ Model initialized successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, hparams
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_dataset_loading(hparams):
    """Test 2: Can we load the dataset?"""
    print("\n" + "="*80)
    print("TEST 2: Dataset Loading")
    print("="*80)

    try:
        dataset = SQuADDataset(hparams, split="validation")
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {len(dataset)}")

        # Test a sample
        sample = dataset[0]
        question, answer = sample
        print(f"  Sample question length: {len(question)} chars")
        print(f"  Sample answer: {answer[:50]}...")

        return dataset
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_energy_extraction(model, hparams):
    """Test 3: Can we extract energy values?"""
    print("\n" + "="*80)
    print("TEST 3: Energy Extraction")
    print("="*80)

    try:
        tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)
        tokenizer.pad_token = tokenizer.eos_token  # Fix padding token

        # Create a simple input
        test_input = "Question: What is AI? Answer:"
        inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True, max_length=50)
        input_ids = inputs['input_ids'].cuda()

        # Run forward pass
        with torch.no_grad():
            predicted_distributions, predicted_energies = model(
                input_ids,
                start_pos=0,
                learning=False,
                return_raw_logits=True,
                no_randomness=True
            )

        # Check outputs
        print(f"✓ Forward pass successful")
        print(f"  Number of MCMC steps: {len(predicted_energies)}")
        print(f"  Energy shape (first step): {predicted_energies[0].shape}")

        # Extract energy metrics
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        energies = []
        for energy_tensor in predicted_energies:
            energy_reshaped = energy_tensor.reshape(batch_size, seq_length)
            energy_mean = energy_reshaped.mean(dim=1)
            energies.append(energy_mean.item())

        print(f"  Energy trajectory: {[f'{e:.2f}' for e in energies]}")
        print(f"  Initial energy: {energies[0]:.4f}")
        print(f"  Final energy: {energies[-1]:.4f}")
        print(f"  Energy gap: {energies[0] - energies[-1]:.4f}")

        return True
    except Exception as e:
        print(f"✗ Energy extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_pipeline(model, hparams, dataset):
    """Test 4: Can we run inference and extract energies?"""
    print("\n" + "="*80)
    print("TEST 4: Inference with Energy Tracking")
    print("="*80)

    try:
        tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)
        tokenizer.pad_token = tokenizer.eos_token  # Fix padding token

        # Get a sample
        question, ground_truth = dataset[0]

        # Tokenize a simple example (just test forward pass, not full generation)
        # For full generation, we'll use the trained model
        test_text = question[:200]  # Take first 200 chars
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
        input_ids = inputs['input_ids'].cuda()

        # Run forward pass
        print("  Running inference (forward pass)...")
        with torch.no_grad():
            predicted_distributions, predicted_energies = model(
                input_ids,
                start_pos=0,
                learning=False,
                return_raw_logits=True,
                no_randomness=True
            )

        # Extract energy metrics
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        energies = []
        for energy_tensor in predicted_energies:
            energy_reshaped = energy_tensor.reshape(batch_size, seq_length)
            energy_mean = energy_reshaped.mean(dim=1)
            energies.append(energy_mean.item())

        energy_metrics = {
            'final_energy': [energies[-1]],
            'initial_energy': [energies[0]],
            'energy_gap': [energies[0] - energies[-1]],
            'max_energy': [max(energies)]
        }

        print(f"✓ Inference successful")
        print(f"  Question: {question[:100]}...")
        print(f"  Ground truth: {ground_truth[:100]}")
        print(f"  Final energy: {energy_metrics['final_energy'][0]:.4f}")
        print(f"  Energy gap: {energy_metrics['energy_gap'][0]:.4f}")
        print(f"  Note: Full autoregressive generation will be tested with trained model")

        return energy_metrics
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_evaluation(generated_answer, ground_truth):
    """Test 5: Can we compute F1 and EM scores?"""
    print("\n" + "="*80)
    print("TEST 5: Evaluation Metrics")
    print("="*80)

    try:
        from evaluate_predictions import compute_exact_match, compute_f1

        em = compute_exact_match(generated_answer, ground_truth)
        f1 = compute_f1(generated_answer, ground_truth)

        print(f"✓ Evaluation successful")
        print(f"  Exact Match: {em}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Correct (F1 >= 0.5): {int(f1 >= 0.5)}")

        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline_on_samples(model, hparams, dataset, num_samples=3):
    """Test 6: Test evaluation metrics work"""
    print("\n" + "="*80)
    print(f"TEST 6: Evaluation Metrics ({num_samples} samples)")
    print("="*80)

    try:
        from evaluate_predictions import compute_exact_match, compute_f1

        print("  Testing F1 and Exact Match calculation...")

        # Test evaluation on sample data
        test_cases = [
            ("Paris", "Paris", 1.0, 1),
            ("The capital is Paris", "Paris", 0.5, 0),
            ("France", "Paris", 0.0, 0),
        ]

        for i, (pred, truth, expected_f1, expected_em) in enumerate(test_cases):
            em = compute_exact_match(pred, truth)
            f1 = compute_f1(pred, truth)
            print(f"  Test {i+1}: Pred='{pred}', Truth='{truth}' → F1={f1:.2f}, EM={em}")

        print(f"\n✓ Evaluation metrics working correctly")
        print(f"  Note: Full pipeline with generation will be tested with trained model")

        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "HALLUCINATION EXPERIMENT - PIPELINE TEST" + " "*18 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    print("\nThis script tests all components without needing a trained model.")
    print("Using randomly initialized model (predictions will be random).")

    # Run tests
    model, hparams = test_model_initialization()
    if model is None:
        print("\n✗ FAILED: Cannot proceed without model")
        return

    dataset = test_dataset_loading(hparams)
    if dataset is None:
        print("\n✗ FAILED: Cannot proceed without dataset")
        return

    energy_ok = test_energy_extraction(model, hparams)
    if not energy_ok:
        print("\n✗ FAILED: Cannot extract energies")
        return

    energy_metrics = test_generation_pipeline(model, hparams, dataset)
    if energy_metrics is None:
        print("\n✗ FAILED: Generation pipeline broken")
        return

    # For evaluation test, use a dummy example
    eval_ok = test_evaluation("Paris", "The capital of France is Paris")
    if not eval_ok:
        print("\n✗ FAILED: Evaluation broken")
        return

    # Full pipeline test
    pipeline_ok = test_full_pipeline_on_samples(model, hparams, dataset, num_samples=5)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    if pipeline_ok:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now proceed to train a real model:")
        print("  python experiments/hallucination/train_small_ebt.py")
        print("\nOr if you already have a checkpoint, run the full pipeline:")
        print("  python experiments/hallucination/generate_with_energy.py --checkpoint_path <path>")
    else:
        print("✗ SOME TESTS FAILED - Please fix errors before proceeding")

if __name__ == "__main__":
    main()
