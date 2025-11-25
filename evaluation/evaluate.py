"""
Evaluation script for AI text detection using HC3 dataset.
Includes statistical analysis and visualization of metrics.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import requests
from datasets import load_dataset
from scipy import stats
from sklearn.metrics import roc_auc_score

API_URL = "http://localhost:8000/api/detect"
N_SAMPLES = 1000

# Metrics to analyze
METRICS = [
    "perplexity",
    "shannon_entropy",
    "burstiness",
    "lexical_diversity",
    "word_length_variance",
    "punctuation_diversity",
    "vocabulary_richness",
    "ml_ai_probability",
    "entropy_ai_probability",
]


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_metric_importance(human_metrics, ai_metrics):
    """Analyze each metric's discriminative power."""
    analysis = {}

    for metric in METRICS:
        human_vals = np.array(human_metrics[metric])
        ai_vals = np.array(ai_metrics[metric])

        # Effect size (Cohen's d)
        cohens_d = abs(calculate_cohens_d(human_vals, ai_vals))

        # Statistical significance
        t_stat, p_value = stats.ttest_ind(human_vals, ai_vals)

        # ROC-AUC (treating higher values as AI prediction)
        labels = np.concatenate([np.zeros(len(human_vals)), np.ones(len(ai_vals))])
        values = np.concatenate([human_vals, ai_vals])
        roc_auc = roc_auc_score(labels, values)
        # Flip if AUC < 0.5 (metric is inverted)
        if roc_auc < 0.5:
            roc_auc = 1 - roc_auc

        # Distribution statistics
        human_mean, human_std = np.mean(human_vals), np.std(human_vals)
        ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)

        analysis[metric] = {
            "cohens_d": cohens_d,
            "roc_auc": roc_auc,
            "p_value": p_value,
            "human_mean": human_mean,
            "human_std": human_std,
            "ai_mean": ai_mean,
            "ai_std": ai_std,
            "separation": abs(human_mean - ai_mean) / ((human_std + ai_std) / 2),
        }

    return analysis


def suggest_weights(analysis):
    """Suggest weights based on statistical analysis."""
    # Normalize scores and combine
    weights = {}

    for metric, metric_stats in analysis.items():
        # Combine effect size and ROC-AUC (both 0-1 range after normalization)
        roc_contribution = (metric_stats["roc_auc"] - 0.5) * 2
        cohens_contribution = min(metric_stats["cohens_d"] / 2.0, 1.0)

        # Weight by both metrics
        weight = 0.5 * roc_contribution + 0.5 * cohens_contribution
        weights[metric] = weight

    # Normalize to sum to 1.0
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


def create_violin_plots(human_metrics, ai_metrics, analysis):
    """Create violin plots for each metric."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Metric Distributions: Human vs AI", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(METRICS):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        human_vals = human_metrics[metric]
        ai_vals = ai_metrics[metric]

        # Create violin plot
        parts = ax.violinplot(
            [human_vals, ai_vals], positions=[1, 2], showmeans=True, showmedians=True
        )

        # Color the violins
        for pc, color in zip(parts["bodies"], ["#66b3ff", "#ff6666"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # Labels and styling
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Human", "AI"])
        title = (
            f"{metric}\n"
            f'(d={analysis[metric]["cohens_d"]:.2f}, '
            f'AUC={analysis[metric]["roc_auc"]:.3f})'
        )
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("metric_distributions.png", dpi=300, bbox_inches="tight")
    print("Saved violin plots to metric_distributions.png")


def main():
    # Load HC3 dataset
    print("Loading HC3 dataset...")
    # HC3 requires trust_remote_code for custom loading script
    dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)

    human_texts = []
    ai_texts = []

    # Collect samples
    for sample in dataset:
        if sample.get("human_answers"):
            for answer in sample["human_answers"]:
                if answer and len(answer) > 50:
                    human_texts.append(answer)

        if sample.get("chatgpt_answers"):
            for answer in sample["chatgpt_answers"]:
                if answer and len(answer) > 50:
                    ai_texts.append(answer)

        if len(human_texts) >= N_SAMPLES and len(ai_texts) >= N_SAMPLES:
            break

    human_texts = human_texts[:N_SAMPLES]
    ai_texts = ai_texts[:N_SAMPLES]

    print(f"Testing {len(human_texts)} human samples and {len(ai_texts)} AI samples...")

    # Collect metrics from API
    human_metrics = {metric: [] for metric in METRICS}
    ai_metrics = {metric: [] for metric in METRICS}

    human_correct = 0
    print("\nTesting human samples...")
    for text in human_texts:
        response = requests.post(API_URL, json={"text": text}, timeout=30)
        data = response.json()

        if data["prediction"] == "human":
            human_correct += 1

        for metric in METRICS:
            human_metrics[metric].append(data[metric])

    ai_correct = 0
    print("Testing AI samples...")
    for text in ai_texts:
        response = requests.post(API_URL, json={"text": text}, timeout=30)
        data = response.json()

        if data["prediction"] == "ai":
            ai_correct += 1

        for metric in METRICS:
            ai_metrics[metric].append(data[metric])

    # Calculate accuracy
    total = len(human_texts) + len(ai_texts)
    accuracy = (human_correct + ai_correct) / total

    print(f"\n{'='*60}")
    print("DETECTION ACCURACY")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Human: {human_correct}/{len(human_texts)} ({human_correct/len(human_texts):.2%})")
    print(f"AI: {ai_correct}/{len(ai_texts)} ({ai_correct/len(ai_texts):.2%})")

    # Statistical analysis
    print(f"\n{'='*60}")
    print("METRIC ANALYSIS")
    print(f"{'='*60}")
    analysis = analyze_metric_importance(human_metrics, ai_metrics)

    # Sort by effect size
    sorted_metrics = sorted(analysis.items(), key=lambda x: x[1]["cohens_d"], reverse=True)

    print(f"\n{'Metric':<25} {'Cohen d':<12} {'ROC-AUC':<12} {'p-value':<12}")
    print("-" * 60)
    for metric, metric_stats in sorted_metrics:
        print(
            f"{metric:<25} {metric_stats['cohens_d']:<12.3f} "
            f"{metric_stats['roc_auc']:<12.3f} "
            f"{metric_stats['p_value']:<12.2e}"
        )

    # Suggest weights
    print(f"\n{'='*60}")
    print("SUGGESTED WEIGHTS (normalized)")
    print(f"{'='*60}")
    weights = suggest_weights(analysis)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    for metric, weight in sorted_weights:
        print(f"{metric:<25} {weight:.4f} ({weight*100:.1f}%)")

    # Create visualizations
    print(f"\n{'='*60}")
    print("Creating violin plots...")
    create_violin_plots(human_metrics, ai_metrics, analysis)

    # Save results
    results = {
        "accuracy": {
            "total_samples": total,
            "overall_accuracy": accuracy,
            "human_correct": human_correct,
            "human_total": len(human_texts),
            "ai_correct": ai_correct,
            "ai_total": len(ai_texts),
        },
        "metric_analysis": analysis,
        "suggested_weights": weights,
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Results saved to results.json")
    print("Visualizations saved to metric_distributions.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
