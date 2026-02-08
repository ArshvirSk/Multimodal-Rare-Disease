"""
Evaluation Module for Multimodal Rare Disease Classifier.
Computes metrics, generates confusion matrices, and compares baselines.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .config import get_config, Config
from .multimodal_classifier import (
    MultimodalClassifier,
    ImageOnlyClassifier,
    TextOnlyClassifier,
)


class Evaluator:
    """
    Evaluator for model performance assessment.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Optional[Config] = None,
        mode: str = "multimodal",
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            test_loader: Test data loader
            config: Configuration object
            mode: Evaluation mode (multimodal, image_only, text_only)
            class_names: List of class names for reporting
        """
        if config is None:
            config = get_config()

        self.config = config
        self.model = model
        self.test_loader = test_loader
        self.mode = mode
        self.class_names = class_names or config.syndrome_names

        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        self.model.eval()

        # Results storage
        self.predictions = []
        self.true_labels = []
        self.probabilities = []

    @torch.no_grad()
    def collect_predictions(self):
        """Collect all predictions from the test set."""
        self.predictions = []
        self.true_labels = []
        self.probabilities = []

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            if self.mode == "multimodal":
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]

                output = self.model(images, input_ids, attention_mask)

            elif self.mode == "image_only":
                images = (
                    batch[0].to(self.device)
                    if isinstance(batch, tuple)
                    else batch["image"].to(self.device)
                )
                labels = batch[1] if isinstance(batch, tuple) else batch["label"]

                output = self.model(images)

            elif self.mode == "text_only":
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]

                output = self.model(input_ids, attention_mask)

            probs = output["probs"].cpu().numpy()
            preds = output["logits"].argmax(dim=-1).cpu().numpy()

            self.predictions.extend(preds)
            self.true_labels.extend(
                labels.numpy() if torch.is_tensor(labels) else labels
            )
            self.probabilities.extend(probs)

        self.predictions = np.array(self.predictions)
        self.true_labels = np.array(self.true_labels)
        self.probabilities = np.array(self.probabilities)

    def compute_metrics(self) -> Dict:
        """
        Compute evaluation metrics.

        Returns:
            Dictionary of metrics
        """
        if len(self.predictions) == 0:
            self.collect_predictions()

        metrics = {}

        # Overall metrics
        metrics["accuracy"] = accuracy_score(self.true_labels, self.predictions)
        metrics["precision_macro"] = precision_score(
            self.true_labels, self.predictions, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            self.true_labels, self.predictions, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            self.true_labels, self.predictions, average="macro", zero_division=0
        )

        # Weighted metrics (accounts for class imbalance)
        metrics["precision_weighted"] = precision_score(
            self.true_labels, self.predictions, average="weighted", zero_division=0
        )
        metrics["recall_weighted"] = recall_score(
            self.true_labels, self.predictions, average="weighted", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            self.true_labels, self.predictions, average="weighted", zero_division=0
        )

        # Per-class metrics
        if self.config.evaluation.per_class_metrics:
            precision_per_class = precision_score(
                self.true_labels, self.predictions, average=None, zero_division=0
            )
            recall_per_class = recall_score(
                self.true_labels, self.predictions, average=None, zero_division=0
            )
            f1_per_class = f1_score(
                self.true_labels, self.predictions, average=None, zero_division=0
            )

            metrics["per_class"] = {}
            for i, class_name in enumerate(
                self.class_names[: len(precision_per_class)]
            ):
                metrics["per_class"][class_name] = {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1": float(f1_per_class[i]),
                }

        # ROC-AUC (for multi-class)
        try:
            if self.probabilities.shape[1] > 2:
                metrics["roc_auc_macro"] = roc_auc_score(
                    self.true_labels,
                    self.probabilities,
                    multi_class="ovr",
                    average="macro",
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    self.true_labels,
                    self.probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix array
        """
        if len(self.predictions) == 0:
            self.collect_predictions()

        return confusion_matrix(self.true_labels, self.predictions)

    def get_classification_report(self) -> str:
        """
        Get detailed classification report.

        Returns:
            Classification report string
        """
        if len(self.predictions) == 0:
            self.collect_predictions()

        return classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names[: len(np.unique(self.true_labels))],
            zero_division=0,
        )

    def plot_confusion_matrix(
        self,
        save_path: Optional[Path] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
    ):
        """
        Plot and optionally save confusion matrix.

        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
            figsize: Figure size
        """
        cm = self.get_confusion_matrix()

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)

        plt.figure(figsize=figsize)

        # Use a subset of class names if needed
        display_names = self.class_names[: cm.shape[0]]

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=display_names,
            yticklabels=display_names,
        )

        plt.title(f"Confusion Matrix ({self.mode})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved confusion matrix to {save_path}")

        plt.close()

    def plot_roc_curves(
        self, save_path: Optional[Path] = None, figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot ROC curves for all classes.

        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if len(self.predictions) == 0:
            self.collect_predictions()

        n_classes = self.probabilities.shape[1]

        plt.figure(figsize=figsize)

        for i in range(min(n_classes, len(self.class_names))):
            # Binary labels for this class
            y_true_binary = (self.true_labels == i).astype(int)
            y_score = self.probabilities[:, i]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{self.class_names[i]} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves ({self.mode})")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved ROC curves to {save_path}")

        plt.close()

    def save_results(self, output_dir: Path):
        """
        Save all evaluation results.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute and save metrics
        metrics = self.compute_metrics()
        metrics_path = output_dir / f"{self.mode}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

        # Save classification report
        report = self.get_classification_report()
        report_path = output_dir / f"{self.mode}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Saved classification report to {report_path}")

        # Save confusion matrix plot
        cm_path = output_dir / f"{self.mode}_confusion_matrix.png"
        self.plot_confusion_matrix(save_path=cm_path)

        # Save ROC curves
        try:
            roc_path = output_dir / f"{self.mode}_roc_curves.png"
            self.plot_roc_curves(save_path=roc_path)
        except Exception as e:
            print(f"Warning: Could not generate ROC curves: {e}")

        # Save predictions if configured
        if self.config.evaluation.save_predictions:
            predictions_path = output_dir / f"{self.mode}_predictions.npz"
            np.savez(
                predictions_path,
                predictions=self.predictions,
                true_labels=self.true_labels,
                probabilities=self.probabilities,
            )
            print(f"Saved predictions to {predictions_path}")


def compare_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    config: Optional[Config] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """
    Compare multiple models on the same test set.

    Args:
        models: Dictionary of {model_name: model}
        test_loader: Test data loader
        config: Configuration object
        output_dir: Directory to save comparison results

    Returns:
        Dictionary of {model_name: metrics}
    """
    if config is None:
        config = get_config()

    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Determine mode from model type
        if isinstance(model, MultimodalClassifier):
            mode = "multimodal"
        elif isinstance(model, ImageOnlyClassifier):
            mode = "image_only"
        elif isinstance(model, TextOnlyClassifier):
            mode = "text_only"
        else:
            mode = "multimodal"

        evaluator = Evaluator(
            model=model, test_loader=test_loader, config=config, mode=mode
        )

        metrics = evaluator.compute_metrics()
        results[name] = metrics

        if output_dir:
            evaluator.save_results(output_dir / name)

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
    print("-" * 60)

    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['accuracy']:.4f}       "
            f"{metrics['f1_macro']:.4f}       "
            f"{metrics['f1_weighted']:.4f}"
        )

    print("=" * 60)

    # Save comparison
    if output_dir:
        comparison_path = output_dir / "model_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved comparison to {comparison_path}")

    return results


def plot_comparison_bar(
    results: Dict[str, Dict],
    metrics: List[str] = ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot bar chart comparing models.

    Args:
        results: Dictionary of {model_name: metrics}
        metrics: List of metrics to compare
        save_path: Path to save plot
        figsize: Figure size
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (name, model_results) in enumerate(results.items()):
        values = [model_results.get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")

    plt.close()


def evaluate_from_checkpoint(
    checkpoint_path: Path,
    test_loader: DataLoader,
    mode: str = "multimodal",
    config: Optional[Config] = None,
) -> Dict:
    """
    Evaluate a model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        test_loader: Test data loader
        mode: Model mode
        config: Configuration object

    Returns:
        Evaluation metrics
    """
    if config is None:
        config = get_config()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create model
    if mode == "multimodal":
        model = MultimodalClassifier(config)
    elif mode == "image_only":
        model = ImageOnlyClassifier(config)
    elif mode == "text_only":
        model = TextOnlyClassifier(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    evaluator = Evaluator(
        model=model, test_loader=test_loader, config=config, mode=mode
    )

    metrics = evaluator.compute_metrics()

    return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate multimodal rare disease classifier"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "image_only", "text_only"],
    )
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    config = get_config()

    print("Evaluation module ready.")
    print("To evaluate, provide a checkpoint and test data loader:")
    print(
        "  python -m src.evaluate --checkpoint checkpoints/multimodal_best.pt --mode multimodal"
    )


if __name__ == "__main__":
    main()
