"""
Comprehensive Evaluation of Multimodal Model
Generates confusion matrix, per-class metrics, and detailed analysis.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.multimodal_classifier import MultimodalClassifier


# Folder name mapping
FOLDER_TO_SYNDROME = {
    "SYN_22Q": "22q11.2 Deletion Syndrome",
    "SYN_AS": "Angelman Syndrome",
    "SYN_CdLS": "Cornelia de Lange Syndrome",
    "SYN_KBG": "KBG Syndrome",
    "SYN_KS": "Kabuki Syndrome",
    "SYN_NBS": "Nicolaides-Baraitser Syndrome",
    "SYN_NS": "Noonan Syndrome",
    "SYN_RSTS": "Rubinstein-Taybi Syndrome",
    "SYN_SMS": "Smith-Magenis Syndrome",
    "SYN_WBS": "Williams-Beuren Syndrome",
    "22q11.2_Deletion_Syndrome": "22q11.2 Deletion Syndrome",
    "Angelman_Syndrome": "Angelman Syndrome",
    "Cornelia_de_Lange_Syndrome": "Cornelia de Lange Syndrome",
    "KBG_Syndrome": "KBG Syndrome",
    "Kabuki_Syndrome": "Kabuki Syndrome",
    "Nicolaides_Baraitser_Syndrome": "Nicolaides-Baraitser Syndrome",
    "Noonan_Syndrome": "Noonan Syndrome",
    "Rubinstein_Taybi_Syndrome": "Rubinstein-Taybi Syndrome",
    "Smith_Magenis_Syndrome": "Smith-Magenis Syndrome",
    "Williams_Beuren_Syndrome": "Williams-Beuren Syndrome",
}


def load_model(checkpoint_path: str = "checkpoints/multimodal_best.pt"):
    """Load trained multimodal model."""
    config = get_config()
    model = MultimodalClassifier(config)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'unknown')}")
    
    return model, config, device


def evaluate_model(
    model,
    config,
    device,
    image_dir: Path,
    clinical_descriptions: dict,
    tokenizer,
    max_samples_per_class: int = None
):
    """
    Evaluate model on all images.
    
    Returns:
        y_true: List of true labels
        y_pred: List of predicted labels
        y_probs: List of probability distributions
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    y_true = []
    y_pred = []
    y_probs = []
    
    syndrome_to_idx = {name: idx for idx, name in enumerate(config.syndrome_names)}
    
    for syndrome_folder in sorted(image_dir.iterdir()):
        if not syndrome_folder.is_dir():
            continue
        
        folder_name = syndrome_folder.name
        true_syndrome = FOLDER_TO_SYNDROME.get(folder_name)
        
        if true_syndrome is None or true_syndrome not in syndrome_to_idx:
            continue
        
        if true_syndrome not in clinical_descriptions:
            continue
        
        true_idx = syndrome_to_idx[true_syndrome]
        clinical_text = clinical_descriptions[true_syndrome]["clinical_description"]
        
        # Get images
        images = list(syndrome_folder.glob("*.jpg")) + list(syndrome_folder.glob("*.png"))
        
        if max_samples_per_class:
            images = images[:max_samples_per_class]
        
        for img_path in images:
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Tokenize text
                encoding = tokenizer(
                    clinical_text,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(
                        images=image_tensor,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                
                probs = output["probs"][0].cpu().numpy()
                pred_idx = probs.argmax()
                
                y_true.append(true_idx)
                y_pred.append(pred_idx)
                y_probs.append(probs)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="results/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Use short names for display
    short_names = [
        "CdLS", "WBS", "Noonan", "Kabuki", "KBG",
        "Angelman", "RTS", "SMS", "NBS", "22q11"
    ]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Multimodal Model - Confusion Matrix", fontsize=14)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(precision, recall, f1, class_names, save_path="results/per_class_metrics.png"):
    """Plot per-class precision, recall, and F1 scores."""
    short_names = [
        "CdLS", "WBS", "Noonan", "Kabuki", "KBG",
        "Angelman", "RTS", "SMS", "NBS", "22q11"
    ]
    
    x = np.arange(len(short_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#2ecc71")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#3498db")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#9b59b6")
    
    ax.set_xlabel("Syndrome", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics - Multimodal Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Per-class metrics saved to {save_path}")


def main():
    print("=" * 70)
    print("MULTIMODAL MODEL EVALUATION")
    print("Comprehensive Metrics and Analysis")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading model...")
    model, config, device = load_model()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder.model_name)
    
    # Load clinical descriptions
    with open("data/syndrome_clinical_descriptions.json", "r") as f:
        clinical_descriptions = json.load(f)
    
    # Find image directory
    image_dir = Path("data/images_augmented")
    if not image_dir.exists():
        image_dir = Path("data/images_organized")
    
    print(f"\n📂 Evaluating on: {image_dir}")
    
    # Evaluate
    print("\n🔍 Running evaluation...")
    y_true, y_pred, y_probs = evaluate_model(
        model, config, device, image_dir, clinical_descriptions, tokenizer
    )
    
    print(f"\n📊 Evaluated {len(y_true)} samples")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"📈 Macro Precision:  {macro_precision * 100:.2f}%")
    print(f"📈 Macro Recall:     {macro_recall * 100:.2f}%")
    print(f"📈 Macro F1-Score:   {macro_f1 * 100:.2f}%")
    
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)
    
    print(f"\n{'Syndrome':<40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)
    
    for i, syndrome in enumerate(config.syndrome_names):
        print(f"{syndrome:<40} {precision[i]*100:>9.1f}% {recall[i]*100:>9.1f}% {f1[i]*100:>9.1f}% {support[i]:>10}")
    
    # Classification report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(
        y_true, y_pred,
        target_names=config.syndrome_names,
        zero_division=0
    ))
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    Path("results").mkdir(exist_ok=True)
    
    plot_confusion_matrix(y_true, y_pred, config.syndrome_names)
    plot_per_class_metrics(precision, recall, f1, config.syndrome_names)
    
    # Save results to JSON
    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "total_samples": len(y_true),
        "per_class": {
            config.syndrome_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(len(config.syndrome_names))
        }
    }
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to results/evaluation_results.json")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - results/confusion_matrix.png")
    print("  - results/per_class_metrics.png")
    print("  - results/evaluation_results.json")


if __name__ == "__main__":
    main()
