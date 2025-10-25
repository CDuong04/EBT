"""
Analyze correlation between energy and hallucinations
Computes AUROC and creates visualizations
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy import stats
import argparse
import os

def load_results(file_path):
    """Load labeled predictions from JSONL"""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def compute_auroc_for_metrics(results, save_dir):
    """
    Compute AUROC for different energy metrics
    Higher energy should correlate with incorrectness
    """
    # Extract labels and energy metrics
    labels = np.array([1 - r['correct'] for r in results])  # 1 = incorrect, 0 = correct

    energy_metrics = {
        'Final Energy': np.array([r['energy_final'] for r in results]),
        'Initial Energy': np.array([r['energy_initial'] for r in results]),
        'Energy Gap': np.array([r['energy_gap'] for r in results]),
        'Max Energy': np.array([r['energy_max'] for r in results]),
    }

    print(f"\n{'='*80}")
    print(f"AUROC ANALYSIS")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Correct: {sum(1 - labels)}, Incorrect: {sum(labels)}")
    print(f"\nAUROC scores (random baseline = 0.5):")
    print(f"{'Metric':<20} {'AUROC':>10} {'Mean (Correct)':>15} {'Mean (Incorrect)':>15}")
    print(f"{'-'*70}")

    auroc_results = {}
    for name, scores in energy_metrics.items():
        # Check if we have both classes
        if len(np.unique(labels)) < 2:
            print(f"{name:<20} {'N/A':>10} (only one class present)")
            continue

        # Compute AUROC
        auroc = roc_auc_score(labels, scores)
        auroc_results[name] = auroc

        # Compute means for correct vs incorrect
        correct_mask = labels == 0
        incorrect_mask = labels == 1
        mean_correct = scores[correct_mask].mean()
        mean_incorrect = scores[incorrect_mask].mean()

        print(f"{name:<20} {auroc:>10.4f} {mean_correct:>15.4f} {mean_incorrect:>15.4f}")

    # Plot ROC curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, scores) in enumerate(energy_metrics.items()):
        if name not in auroc_results:
            continue

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = auroc_results[name]

        # Plot
        axes[idx].plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}')
        axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[idx].set_xlabel('False Positive Rate', fontsize=11)
        axes[idx].set_ylabel('True Positive Rate', fontsize=11)
        axes[idx].set_title(f'ROC Curve: {name}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    roc_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=150)
    print(f"\nROC curves saved to {roc_path}")
    plt.close()

    return auroc_results, energy_metrics, labels

def plot_energy_distributions(energy_metrics, labels, save_dir):
    """Plot energy distributions for correct vs incorrect predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, scores) in enumerate(energy_metrics.items()):
        correct_scores = scores[labels == 0]
        incorrect_scores = scores[labels == 1]

        # Plot histograms
        axes[idx].hist(correct_scores, bins=30, alpha=0.6, label='Correct', color='green', density=True)
        axes[idx].hist(incorrect_scores, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)

        axes[idx].set_xlabel(name, fontsize=11)
        axes[idx].set_ylabel('Density', fontsize=11)
        axes[idx].set_title(f'Distribution: {name}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)

        # Add statistics
        t_stat, p_value = stats.ttest_ind(correct_scores, incorrect_scores)
        axes[idx].text(0.05, 0.95, f't-test p-value: {p_value:.4f}',
                      transform=axes[idx].transAxes, fontsize=9,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'energy_distributions.png')
    plt.savefig(dist_path, dpi=150)
    print(f"Energy distributions saved to {dist_path}")
    plt.close()

def analyze_rejection_threshold(energy_metrics, labels, save_dir):
    """
    Analyze what happens if we reject predictions based on energy threshold
    """
    print(f"\n{'='*80}")
    print(f"REJECTION ANALYSIS")
    print(f"{'='*80}")

    results_text = []

    for name, scores in energy_metrics.items():
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Calculate metrics at different rejection rates
        print(f"\n{name}:")
        print(f"  Optimal threshold: {optimal_threshold:.4f}")

        header = f"{'Rejection %':<15} {'Precision':>12} {'Recall':>12} {'Errors Caught':>15}"
        print(f"  {header}")
        print(f"  {'-'*60}")

        for rejection_pct in [10, 20, 30, 40, 50]:
            # Reject top X% highest energy predictions
            threshold = np.percentile(scores, 100 - rejection_pct)
            rejected_mask = scores >= threshold

            # Of rejected predictions, how many were incorrect?
            rejected_incorrect = np.sum(rejected_mask & (labels == 1))
            total_rejected = np.sum(rejected_mask)
            total_incorrect = np.sum(labels == 1)

            precision = rejected_incorrect / total_rejected if total_rejected > 0 else 0
            recall = rejected_incorrect / total_incorrect if total_incorrect > 0 else 0

            print(f"  {rejection_pct:>2}%{'':<12} {precision:>12.2%} {recall:>12.2%} {rejected_incorrect:>6}/{total_incorrect:<6}")

        results_text.append(f"\n{name}: Optimal threshold = {optimal_threshold:.4f}")

    # Plot precision-recall curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, scores) in enumerate(energy_metrics.items()):
        precision, recall, thresholds = precision_recall_curve(labels, scores)

        axes[idx].plot(recall, precision, linewidth=2)
        axes[idx].set_xlabel('Recall (Errors Caught)', fontsize=11)
        axes[idx].set_ylabel('Precision (Rejection Accuracy)', fontsize=11)
        axes[idx].set_title(f'Precision-Recall: {name}', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    pr_path = os.path.join(save_dir, 'precision_recall_curves.png')
    plt.savefig(pr_path, dpi=150)
    print(f"\nPrecision-Recall curves saved to {pr_path}")
    plt.close()

def create_scatter_plot(results, save_dir):
    """Create scatter plot of energy vs F1 score"""
    f1_scores = np.array([r['f1'] for r in results])
    final_energies = np.array([r['energy_final'] for r in results])
    correct = np.array([r['correct'] for r in results])

    plt.figure(figsize=(10, 6))

    # Plot correct and incorrect separately
    plt.scatter(final_energies[correct == 1], f1_scores[correct == 1],
               alpha=0.5, c='green', label='Correct (F1 ≥ 0.5)', s=30)
    plt.scatter(final_energies[correct == 0], f1_scores[correct == 0],
               alpha=0.5, c='red', label='Incorrect (F1 < 0.5)', s=30)

    plt.xlabel('Final Energy', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Energy vs F1 Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Add correlation
    corr, p_value = stats.pearsonr(final_energies, f1_scores)
    plt.text(0.05, 0.95, f'Pearson r = {corr:.3f} (p = {p_value:.4f})',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top', fontsize=10)

    plt.tight_layout()
    scatter_path = os.path.join(save_dir, 'energy_vs_f1_scatter.png')
    plt.savefig(scatter_path, dpi=150)
    print(f"Scatter plot saved to {scatter_path}")
    plt.close()

def generate_report(auroc_results, save_dir):
    """Generate a summary report"""
    report_path = os.path.join(save_dir, 'analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENERGY-BASED HALLUCINATION DETECTION - ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("AUROC SCORES:\n")
        f.write("-" * 40 + "\n")
        for metric, auroc in sorted(auroc_results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {metric:<20}: {auroc:.4f}\n")

        f.write("\n\nINTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        f.write("AUROC = 0.5: Random (no predictive power)\n")
        f.write("AUROC > 0.6: Weak correlation\n")
        f.write("AUROC > 0.7: Moderate correlation\n")
        f.write("AUROC > 0.8: Strong correlation\n")

        best_metric = max(auroc_results.items(), key=lambda x: x[1])
        f.write(f"\n\nBEST METRIC: {best_metric[0]} (AUROC = {best_metric[1]:.4f})\n")

        if best_metric[1] > 0.6:
            f.write("\n✓ SUCCESS: Energy shows correlation with hallucinations!\n")
            f.write("  Recommend proceeding to Phase 2 (intervention experiments)\n")
        else:
            f.write("\n✗ WEAK SIGNAL: Energy shows limited correlation\n")
            f.write("  Recommend trying:\n")
            f.write("  - Different energy aggregation methods\n")
            f.write("  - More MCMC steps during inference\n")
            f.write("  - Contrastive fine-tuning on correct/incorrect examples\n")

    print(f"\nAnalysis report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='experiments/hallucination/results/predictions_labeled.jsonl',
                        help='Input JSONL file with labeled predictions')
    parser.add_argument('--output_dir', type=str,
                        default='experiments/hallucination/results/analysis',
                        help='Output directory for plots and report')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from {args.input_file}...")
    results = load_results(args.input_file)

    # Compute AUROC for different metrics
    auroc_results, energy_metrics, labels = compute_auroc_for_metrics(results, args.output_dir)

    # Plot energy distributions
    plot_energy_distributions(energy_metrics, labels, args.output_dir)

    # Analyze rejection thresholds
    analyze_rejection_threshold(energy_metrics, labels, args.output_dir)

    # Create scatter plot
    create_scatter_plot(results, args.output_dir)

    # Generate summary report
    generate_report(auroc_results, args.output_dir)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"All results saved to: {args.output_dir}")
    print(f"\nFiles generated:")
    print(f"  - roc_curves.png")
    print(f"  - energy_distributions.png")
    print(f"  - precision_recall_curves.png")
    print(f"  - energy_vs_f1_scatter.png")
    print(f"  - analysis_report.txt")

if __name__ == "__main__":
    main()
