"""
Evaluate predictions: compute Exact Match and F1 scores
Adds correctness labels to the predictions file
"""
import json
import re
import string
from collections import Counter
import argparse

def normalize_answer(s):
    """
    Normalize answer string (from official SQuAD evaluation script)
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    """
    Compute exact match score
    Returns 1 if normalized strings match, 0 otherwise
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    """
    Compute F1 score between prediction and ground truth
    Based on token overlap
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # If either is empty, check for exact match
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    # Count common tokens
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def evaluate_file(input_file, output_file, f1_threshold=0.5):
    """
    Read predictions JSONL, compute metrics, save with correctness labels
    """
    results = []
    total_em = 0
    total_f1 = 0
    total_correct_by_f1 = 0

    print(f"Reading predictions from {input_file}")

    with open(input_file, 'r') as f:
        for line in f:
            result = json.loads(line)

            # Compute metrics
            em = compute_exact_match(result['generated_answer'], result['ground_truth'])
            f1 = compute_f1(result['generated_answer'], result['ground_truth'])

            # Add to result
            result['exact_match'] = em
            result['f1'] = f1
            result['correct'] = int(f1 >= f1_threshold)  # Binary label based on F1 threshold

            results.append(result)

            # Aggregate stats
            total_em += em
            total_f1 += f1
            total_correct_by_f1 += result['correct']

    num_samples = len(results)

    # Compute overall metrics
    avg_em = total_em / num_samples * 100
    avg_f1 = total_f1 / num_samples * 100
    accuracy = total_correct_by_f1 / num_samples * 100

    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples: {num_samples}")
    print(f"Average Exact Match: {avg_em:.2f}%")
    print(f"Average F1 Score: {avg_f1:.2f}%")
    print(f"Accuracy (F1 >= {f1_threshold}): {accuracy:.2f}%")
    print(f"Correct: {total_correct_by_f1}, Incorrect: {num_samples - total_correct_by_f1}")

    # Save results with correctness labels
    print(f"\nSaving labeled results to {output_file}")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Done! Results saved.")

    # Show some examples
    print(f"\n{'='*80}")
    print(f"SAMPLE PREDICTIONS")
    print(f"{'='*80}")

    # Show some correct and incorrect examples
    correct_examples = [r for r in results if r['correct'] == 1][:2]
    incorrect_examples = [r for r in results if r['correct'] == 0][:2]

    print("\nCORRECT EXAMPLES:")
    for i, ex in enumerate(correct_examples):
        print(f"\nExample {i+1}:")
        print(f"Question: {ex['question'][:150]}...")
        print(f"Generated: {ex['generated_answer']}")
        print(f"Ground Truth: {ex['ground_truth']}")
        print(f"F1: {ex['f1']:.2f}, EM: {ex['exact_match']}")
        print(f"Energy (final): {ex['energy_final']:.4f}, Gap: {ex['energy_gap']:.4f}")

    print("\nINCORRECT EXAMPLES:")
    for i, ex in enumerate(incorrect_examples):
        print(f"\nExample {i+1}:")
        print(f"Question: {ex['question'][:150]}...")
        print(f"Generated: {ex['generated_answer']}")
        print(f"Ground Truth: {ex['ground_truth']}")
        print(f"F1: {ex['f1']:.2f}, EM: {ex['exact_match']}")
        print(f"Energy (final): {ex['energy_final']:.4f}, Gap: {ex['energy_gap']:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='experiments/hallucination/results/predictions_with_energy.jsonl',
                        help='Input JSONL file with predictions and energies')
    parser.add_argument('--output_file', type=str,
                        default='experiments/hallucination/results/predictions_labeled.jsonl',
                        help='Output JSONL file with correctness labels')
    parser.add_argument('--f1_threshold', type=float, default=0.5,
                        help='F1 threshold for binary correctness label')
    args = parser.parse_args()

    evaluate_file(args.input_file, args.output_file, args.f1_threshold)

if __name__ == "__main__":
    main()
