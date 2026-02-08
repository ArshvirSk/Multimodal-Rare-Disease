"""
Chi-Square Statistical Test for Model Comparison.
Validates that multimodal model improvement is statistically significant.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mcnemar


def chi_square_test(
    predictions_model1: np.ndarray,
    predictions_model2: np.ndarray,
    true_labels: np.ndarray,
) -> Dict:
    """
    Perform chi-square test comparing two models' predictions.

    Tests whether there is a significant association between
    model type and prediction correctness.

    Args:
        predictions_model1: Predictions from model 1
        predictions_model2: Predictions from model 2
        true_labels: Ground truth labels

    Returns:
        Dictionary with test results
    """
    # Create correctness arrays
    correct1 = (predictions_model1 == true_labels).astype(int)
    correct2 = (predictions_model2 == true_labels).astype(int)

    # Create contingency table
    # Rows: Model (model1, model2)
    # Cols: Correctness (incorrect, correct)
    contingency_table = np.array(
        [
            [np.sum(correct1 == 0), np.sum(correct1 == 1)],
            [np.sum(correct2 == 0), np.sum(correct2 == 1)],
        ]
    )

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Calculate accuracy difference
    acc1 = np.mean(correct1)
    acc2 = np.mean(correct2)

    results = {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "contingency_table": contingency_table.tolist(),
        "expected_frequencies": expected.tolist(),
        "model1_accuracy": float(acc1),
        "model2_accuracy": float(acc2),
        "accuracy_difference": float(acc1 - acc2),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "significant_at_0.001": p_value < 0.001,
    }

    return results


def mcnemar_test(
    predictions_model1: np.ndarray,
    predictions_model2: np.ndarray,
    true_labels: np.ndarray,
) -> Dict:
    """
    Perform McNemar's test comparing two models on paired samples.

    McNemar's test is more appropriate than chi-square for paired data
    (same samples evaluated by both models).

    Args:
        predictions_model1: Predictions from model 1
        predictions_model2: Predictions from model 2
        true_labels: Ground truth labels

    Returns:
        Dictionary with test results
    """
    # Create correctness arrays
    correct1 = (predictions_model1 == true_labels).astype(int)
    correct2 = (predictions_model2 == true_labels).astype(int)

    # Create contingency table for McNemar's test
    # b = model1 correct, model2 incorrect
    # c = model1 incorrect, model2 correct
    b = np.sum((correct1 == 1) & (correct2 == 0))
    c = np.sum((correct1 == 0) & (correct2 == 1))
    a = np.sum((correct1 == 1) & (correct2 == 1))
    d = np.sum((correct1 == 0) & (correct2 == 0))

    contingency_table = np.array([[a, b], [c, d]])

    # Perform McNemar's test
    # Using exact binomial test when b + c < 25
    if b + c < 25:
        # Exact binomial test
        result = stats.binomtest(b, b + c, 0.5)
        p_value = result.pvalue
        test_type = "exact_binomial"
    else:
        # McNemar's chi-square approximation
        result = mcnemar(contingency_table, exact=False, correction=True)
        p_value = result.pvalue
        test_type = "mcnemar_chi2"

    # Calculate accuracy difference
    acc1 = np.mean(correct1)
    acc2 = np.mean(correct2)

    results = {
        "test_type": test_type,
        "p_value": float(p_value),
        "contingency_table": {
            "both_correct": int(a),
            "model1_only_correct": int(b),
            "model2_only_correct": int(c),
            "both_incorrect": int(d),
        },
        "discordant_pairs": int(b + c),
        "model1_accuracy": float(acc1),
        "model2_accuracy": float(acc2),
        "accuracy_difference": float(acc1 - acc2),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "significant_at_0.001": p_value < 0.001,
    }

    return results


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    metric: str = "accuracy",
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        predictions: Model predictions
        true_labels: Ground truth labels
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metric: Metric to compute (accuracy, f1, etc.)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    n_samples = len(predictions)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_pred = predictions[indices]
        boot_true = true_labels[indices]

        if metric == "accuracy":
            score = np.mean(boot_pred == boot_true)
        else:
            score = np.mean(boot_pred == boot_true)  # Fallback to accuracy

        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    point_estimate = np.mean(predictions == true_labels)

    return point_estimate, lower, upper


def compare_multimodal_vs_unimodal(
    multimodal_predictions: np.ndarray,
    image_only_predictions: np.ndarray,
    text_only_predictions: np.ndarray,
    true_labels: np.ndarray,
) -> Dict:
    """
    Compare multimodal model against both unimodal baselines.

    Args:
        multimodal_predictions: Predictions from multimodal model
        image_only_predictions: Predictions from image-only model
        text_only_predictions: Predictions from text-only model
        true_labels: Ground truth labels

    Returns:
        Dictionary with all comparison results
    """
    results = {
        "multimodal_vs_image_only": {},
        "multimodal_vs_text_only": {},
        "image_only_vs_text_only": {},
        "summary": {},
    }

    # Multimodal vs Image-only
    print("\n--- Multimodal vs Image-Only ---")
    chi2_result = chi_square_test(
        multimodal_predictions, image_only_predictions, true_labels
    )
    mcnemar_result = mcnemar_test(
        multimodal_predictions, image_only_predictions, true_labels
    )
    results["multimodal_vs_image_only"] = {
        "chi_square": chi2_result,
        "mcnemar": mcnemar_result,
    }
    print(f"Chi-square p-value: {chi2_result['p_value']:.6f}")
    print(f"McNemar p-value: {mcnemar_result['p_value']:.6f}")
    print(f"Accuracy difference: {chi2_result['accuracy_difference']:.4f}")
    print(f"Significant (p<0.05): {mcnemar_result['significant_at_0.05']}")

    # Multimodal vs Text-only
    print("\n--- Multimodal vs Text-Only ---")
    chi2_result = chi_square_test(
        multimodal_predictions, text_only_predictions, true_labels
    )
    mcnemar_result = mcnemar_test(
        multimodal_predictions, text_only_predictions, true_labels
    )
    results["multimodal_vs_text_only"] = {
        "chi_square": chi2_result,
        "mcnemar": mcnemar_result,
    }
    print(f"Chi-square p-value: {chi2_result['p_value']:.6f}")
    print(f"McNemar p-value: {mcnemar_result['p_value']:.6f}")
    print(f"Accuracy difference: {chi2_result['accuracy_difference']:.4f}")
    print(f"Significant (p<0.05): {mcnemar_result['significant_at_0.05']}")

    # Image-only vs Text-only
    print("\n--- Image-Only vs Text-Only ---")
    chi2_result = chi_square_test(
        image_only_predictions, text_only_predictions, true_labels
    )
    mcnemar_result = mcnemar_test(
        image_only_predictions, text_only_predictions, true_labels
    )
    results["image_only_vs_text_only"] = {
        "chi_square": chi2_result,
        "mcnemar": mcnemar_result,
    }
    print(f"Chi-square p-value: {chi2_result['p_value']:.6f}")
    print(f"McNemar p-value: {mcnemar_result['p_value']:.6f}")

    # Summary
    multimodal_acc = np.mean(multimodal_predictions == true_labels)
    image_acc = np.mean(image_only_predictions == true_labels)
    text_acc = np.mean(text_only_predictions == true_labels)

    results["summary"] = {
        "multimodal_accuracy": float(multimodal_acc),
        "image_only_accuracy": float(image_acc),
        "text_only_accuracy": float(text_acc),
        "multimodal_improvement_over_image": float(multimodal_acc - image_acc),
        "multimodal_improvement_over_text": float(multimodal_acc - text_acc),
        "best_unimodal": "image" if image_acc > text_acc else "text",
        "multimodal_beats_both": multimodal_acc > max(image_acc, text_acc),
    }

    # Bootstrap confidence intervals
    print("\n--- Bootstrap Confidence Intervals (95%) ---")
    mm_est, mm_low, mm_high = bootstrap_confidence_interval(
        multimodal_predictions, true_labels
    )
    img_est, img_low, img_high = bootstrap_confidence_interval(
        image_only_predictions, true_labels
    )
    txt_est, txt_low, txt_high = bootstrap_confidence_interval(
        text_only_predictions, true_labels
    )

    print(f"Multimodal: {mm_est:.4f} [{mm_low:.4f}, {mm_high:.4f}]")
    print(f"Image-only: {img_est:.4f} [{img_low:.4f}, {img_high:.4f}]")
    print(f"Text-only:  {txt_est:.4f} [{txt_low:.4f}, {txt_high:.4f}]")

    results["confidence_intervals"] = {
        "multimodal": {"estimate": mm_est, "lower": mm_low, "upper": mm_high},
        "image_only": {"estimate": img_est, "lower": img_low, "upper": img_high},
        "text_only": {"estimate": txt_est, "lower": txt_low, "upper": txt_high},
    }

    return results


def run_statistical_validation(
    predictions_dir: Path, output_path: Optional[Path] = None
) -> Dict:
    """
    Run statistical validation from saved prediction files.

    Args:
        predictions_dir: Directory containing prediction .npz files
        output_path: Path to save results

    Returns:
        Statistical comparison results
    """
    predictions_dir = Path(predictions_dir)

    # Load predictions
    files = {
        "multimodal": predictions_dir / "multimodal_predictions.npz",
        "image_only": predictions_dir / "image_only_predictions.npz",
        "text_only": predictions_dir / "text_only_predictions.npz",
    }

    loaded_data = {}
    for name, path in files.items():
        if path.exists():
            data = np.load(path)
            loaded_data[name] = {
                "predictions": data["predictions"],
                "true_labels": data["true_labels"],
            }
            print(f"Loaded {name} predictions: {len(data['predictions'])} samples")
        else:
            print(f"Warning: {path} not found")

    # Validate all files are loaded
    required = ["multimodal", "image_only", "text_only"]
    if not all(name in loaded_data for name in required):
        print("Error: Not all prediction files found")
        return {}

    # Verify labels match
    true_labels = loaded_data["multimodal"]["true_labels"]
    for name in ["image_only", "text_only"]:
        if not np.array_equal(true_labels, loaded_data[name]["true_labels"]):
            print(f"Warning: True labels mismatch for {name}")

    # Run comparison
    results = compare_multimodal_vs_unimodal(
        multimodal_predictions=loaded_data["multimodal"]["predictions"],
        image_only_predictions=loaded_data["image_only"]["predictions"],
        text_only_predictions=loaded_data["text_only"]["predictions"],
        true_labels=true_labels,
    )

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved statistical results to {output_path}")

    return results


def print_hypothesis_conclusion(results: Dict):
    """
    Print conclusion about the hypothesis test.

    Args:
        results: Statistical comparison results
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST CONCLUSION")
    print("=" * 60)

    print(
        "\nNull Hypothesis (H0): Multimodal and unimodal models have same performance"
    )
    print(
        "Alternative Hypothesis (H1): Multimodal model outperforms unimodal baselines"
    )
    print()

    # Check multimodal vs image-only
    mm_vs_img = results.get("multimodal_vs_image_only", {}).get("mcnemar", {})
    mm_vs_txt = results.get("multimodal_vs_text_only", {}).get("mcnemar", {})
    summary = results.get("summary", {})

    if (
        mm_vs_img.get("significant_at_0.05")
        and mm_vs_img.get("accuracy_difference", 0) > 0
    ):
        print("✓ Multimodal SIGNIFICANTLY outperforms Image-Only (p < 0.05)")
    else:
        print("✗ No significant difference between Multimodal and Image-Only")

    if (
        mm_vs_txt.get("significant_at_0.05")
        and mm_vs_txt.get("accuracy_difference", 0) > 0
    ):
        print("✓ Multimodal SIGNIFICANTLY outperforms Text-Only (p < 0.05)")
    else:
        print("✗ No significant difference between Multimodal and Text-Only")

    print()
    if summary.get("multimodal_beats_both"):
        print("CONCLUSION: Multimodal fusion provides measurable improvement.")
        improvement_img = summary.get("multimodal_improvement_over_image", 0) * 100
        improvement_txt = summary.get("multimodal_improvement_over_text", 0) * 100
        print(f"  - +{improvement_img:.1f}% over image-only baseline")
        print(f"  - +{improvement_txt:.1f}% over text-only baseline")
    else:
        print("CONCLUSION: Multimodal does not outperform both unimodal baselines.")

    print("=" * 60)


def main():
    """Main entry point for chi-square testing."""
    parser = argparse.ArgumentParser(description="Chi-square test for model comparison")
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="results",
        help="Directory containing prediction .npz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/statistical_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run demo with synthetic data"
    )

    args = parser.parse_args()

    if args.demo:
        print("Running demo with synthetic predictions...")

        # Generate synthetic predictions
        np.random.seed(42)
        n_samples = 500
        n_classes = 10

        true_labels = np.random.randint(0, n_classes, n_samples)

        # Simulate different model accuracies
        # Multimodal: ~85% accuracy
        multimodal_pred = true_labels.copy()
        noise_idx = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
        multimodal_pred[noise_idx] = np.random.randint(0, n_classes, len(noise_idx))

        # Image-only: ~75% accuracy
        image_pred = true_labels.copy()
        noise_idx = np.random.choice(n_samples, int(n_samples * 0.25), replace=False)
        image_pred[noise_idx] = np.random.randint(0, n_classes, len(noise_idx))

        # Text-only: ~70% accuracy
        text_pred = true_labels.copy()
        noise_idx = np.random.choice(n_samples, int(n_samples * 0.30), replace=False)
        text_pred[noise_idx] = np.random.randint(0, n_classes, len(noise_idx))

        # Run comparison
        results = compare_multimodal_vs_unimodal(
            multimodal_pred, image_pred, text_pred, true_labels
        )

        print_hypothesis_conclusion(results)

        # Save results
        with open("results/demo_statistical_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved demo results to results/demo_statistical_results.json")
    else:
        results = run_statistical_validation(
            predictions_dir=Path(args.predictions_dir), output_path=Path(args.output)
        )

        if results:
            print_hypothesis_conclusion(results)


if __name__ == "__main__":
    main()
