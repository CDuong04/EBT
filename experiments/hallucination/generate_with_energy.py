"""
Generate answers on SQuAD validation set and track energy values
Saves results in JSONL format for analysis
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.nlp.squad_dataloader import SQuADDataset
from data.nlp.collator import NLP_HF_Collator
from model.nlp.ebt import EBT_NLP

class InferenceHParams:
    """Wrapper to convert dict to object with attributes"""
    def __init__(self, hparams_dict):
        for key, value in hparams_dict.items():
            setattr(self, key, value)

def extract_energies_from_forward(model, input_ids):
    """
    Run model forward pass and extract energy values
    Returns: final_energy, initial_energy, energy_gap, all_energies
    """
    with torch.no_grad():
        # Run forward pass (returns predicted_distributions, predicted_energies)
        predicted_distributions, predicted_energies = model(
            input_ids,
            start_pos=0,
            learning=False,
            return_raw_logits=True,
            no_randomness=True
        )

    # predicted_energies is a list of tensors, one per MCMC step
    # Each tensor has shape [batch_size * seq_length, 1]

    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]

    # Reshape and aggregate energies
    all_energies = []
    for energy_tensor in predicted_energies:
        # Reshape to [batch_size, seq_length]
        energy_reshaped = energy_tensor.reshape(batch_size, seq_length)
        # Average over sequence dimension for batch
        energy_mean = energy_reshaped.mean(dim=1)  # [batch_size]
        all_energies.append(energy_mean)

    # Stack all MCMC steps: [num_mcmc_steps, batch_size]
    all_energies = torch.stack(all_energies)

    # Extract metrics
    initial_energy = all_energies[0]  # First MCMC step
    final_energy = all_energies[-1]   # Last MCMC step
    energy_gap = initial_energy - final_energy
    max_energy = all_energies.max(dim=0)[0]

    return {
        'final_energy': final_energy.detach().cpu().numpy().tolist(),
        'initial_energy': initial_energy.detach().cpu().numpy().tolist(),
        'energy_gap': energy_gap.detach().cpu().numpy().tolist(),
        'max_energy': max_energy.detach().cpu().numpy().tolist(),
        'all_energies': all_energies.detach().cpu().numpy().tolist()  # [num_mcmc_steps, batch_size]
    }

def generate_answer_greedy(model, tokenizer, context_and_question, max_gen_len=50, max_context_len=256):
    """
    Greedy generation with energy tracking for EBT model
    Returns: generated_text, energy_metrics
    """
    # Tokenize input
    inputs = tokenizer(
        context_and_question,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_len
    )

    prompt_tokens = inputs['input_ids'][0].tolist()
    prompt_length = len(prompt_tokens)

    # Total length for generation buffer
    max_total_len = min(512, prompt_length + max_gen_len)  # Stay within reasonable bounds

    # Initialize token buffer
    tokens = torch.full((1, max_total_len), tokenizer.eos_token_id, dtype=torch.long, device="cuda")
    tokens[0, :prompt_length] = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")

    # Track energies during generation
    all_step_energies = []

    with torch.no_grad():
        # Generate tokens one by one
        for cur_pos in range(prompt_length, max_total_len):
            # Get current sequence (from beginning to cur_pos)
            input_tokens = tokens[:, :cur_pos]

            # Call model
            predicted_distributions, predicted_energies = model(
                input_tokens,
                start_pos=0,
                learning=False,
                return_raw_logits=True,
                no_randomness=True
            )

            # Get final MCMC step logits
            logits = predicted_distributions[-1]  # [1, seq_length, vocab_size]

            # Get next token (greedy - take argmax)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            tokens[0, cur_pos] = next_token

            # Track energy for this position
            final_energies = predicted_energies[-1]  # [seq_length, 1]
            position_energy = final_energies[-1]  # Energy of last position
            all_step_energies.append(position_energy.detach().cpu().item())

            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode generated text (only the answer part)
    generated_tokens = tokens[0, prompt_length:cur_pos].tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Compute energy metrics for the final sequence
    final_sequence = tokens[:, :cur_pos]
    energy_metrics = extract_energies_from_forward(model, final_sequence)

    # Add per-token energies during generation
    energy_metrics['generation_step_energies'] = all_step_energies

    return generated_text, energy_metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained EBT checkpoint')
    parser.add_argument('--output_file', type=str, default='experiments/hallucination/results/predictions_with_energy.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of validation samples to evaluate')
    parser.add_argument('--max_gen_len', type=int, default=50,
                        help='Maximum generation length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (keep at 1 for simplicity)')
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda')

    # Get hyperparameters from checkpoint
    hparams = checkpoint['hyper_parameters']
    hparams['execution_mode'] = 'inference'  # Override to inference mode
    hparams['max_seq_len'] = 2048  # Increase max sequence length for generation
    hparams_obj = InferenceHParams(hparams)

    # Initialize model
    print("Initializing model...")
    model = EBT_NLP(hparams_obj)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    # Recompute freqs_cis with the new max_seq_len to avoid sequence length issues
    from model.ar_ebt_default import precompute_freqs_cis
    model.transformer.freqs_cis = precompute_freqs_cis(
        model.transformer.params.dim // model.transformer.params.n_heads,
        hparams['max_seq_len']
    ).to(model.device)

    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hparams['tokenizer'], clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    # Load dataset
    print("Loading SQuAD validation dataset...")
    dataset = SQuADDataset(hparams_obj, split="validation")

    # Limit number of samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating on {num_samples} samples")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Generate predictions
    results = []
    with open(args.output_file, 'w') as f:
        for i in tqdm(range(num_samples), desc="Generating answers"):
            # Get sample
            context_and_question, ground_truth = dataset[i]

            # Generate answer with energy tracking
            generated_answer, energy_metrics = generate_answer_greedy(
                model, tokenizer, context_and_question, max_gen_len=args.max_gen_len
            )

            # Prepare result
            result = {
                'idx': i,
                'question': context_and_question,
                'generated_answer': generated_answer,
                'ground_truth': ground_truth,
                'energy_final': energy_metrics['final_energy'][0],
                'energy_initial': energy_metrics['initial_energy'][0],
                'energy_gap': energy_metrics['energy_gap'][0],
                'energy_max': energy_metrics['max_energy'][0],
                'all_energies': energy_metrics['all_energies'],
            }

            # Write to JSONL
            f.write(json.dumps(result) + '\n')
            results.append(result)

    print(f"\nResults saved to {args.output_file}")
    print(f"Total samples: {len(results)}")

    # Print some sample outputs
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS:")
    print("="*80)
    for i in range(min(3, len(results))):
        print(f"\nSample {i+1}:")
        print(f"Question: {results[i]['question'][:200]}...")
        print(f"Generated: {results[i]['generated_answer']}")
        print(f"Ground Truth: {results[i]['ground_truth']}")
        print(f"Energy (final): {results[i]['energy_final']:.4f}")
        print(f"Energy (gap): {results[i]['energy_gap']:.4f}")

if __name__ == "__main__":
    main()
