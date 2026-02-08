"""
Rare Disease Prediction Demo Script
====================================
Use this script to predict the syndrome from a facial image.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model checkpoints/best_model.pt
"""

import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.multimodal_classifier import ImageOnlyClassifier


# Syndrome names in order
SYNDROME_NAMES = [
    "Cornelia de Lange Syndrome",
    "Williams-Beuren Syndrome", 
    "Noonan Syndrome",
    "Kabuki Syndrome",
    "KBG Syndrome",
    "Angelman Syndrome",
    "Rubinstein-Taybi Syndrome",
    "Smith-Magenis Syndrome",
    "Nicolaides-Baraitser Syndrome",
    "22q11.2 Deletion Syndrome",
]


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    config = get_config()
    model = ImageOnlyClassifier(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(model, image_tensor: torch.Tensor, device: str = "cuda") -> dict:
    """Run prediction on preprocessed image."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        logits = output["logits"]
        probs = torch.softmax(logits, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probs, k=min(5, len(SYNDROME_NAMES)))
        
    results = {
        "predictions": [],
        "top_prediction": {
            "syndrome": SYNDROME_NAMES[top_indices[0][0].item()],
            "confidence": top_probs[0][0].item() * 100
        }
    }
    
    for i in range(len(top_indices[0])):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item() * 100
        results["predictions"].append({
            "rank": i + 1,
            "syndrome": SYNDROME_NAMES[idx],
            "confidence": prob
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict rare genetic syndrome from facial image"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input facial image"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Please train a model first using: python run_training.py")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("RARE DISEASE SYNDROME PREDICTION")
    print(f"{'='*60}")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, args.device)
    
    # Preprocess image
    print("Processing image...")
    image_tensor = preprocess_image(args.image)
    
    # Predict
    print("Running prediction...\n")
    results = predict(model, image_tensor, args.device)
    
    # Display results
    print(f"{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"\n🎯 Top Prediction: {results['top_prediction']['syndrome']}")
    print(f"   Confidence: {results['top_prediction']['confidence']:.1f}%\n")
    
    print("Top 5 Predictions:")
    print("-" * 50)
    for pred in results["predictions"]:
        bar = "█" * int(pred["confidence"] / 5) + "░" * (20 - int(pred["confidence"] / 5))
        print(f"  {pred['rank']}. {pred['syndrome'][:35]:<35}")
        print(f"     {bar} {pred['confidence']:.1f}%")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    main()
