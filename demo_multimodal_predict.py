"""
Demo: Multimodal Prediction with Image + Clinical Text
Shows how the model uses BOTH modalities for classification.
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.multimodal_classifier import MultimodalClassifier


def load_model(checkpoint_path: str = "checkpoints/multimodal_best.pt"):
    """Load trained multimodal model."""
    config = get_config()
    model = MultimodalClassifier(config)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, config, device


def predict_multimodal(
    model,
    config,
    device,
    image_path: str,
    clinical_text: str,
    tokenizer
):
    """
    Make prediction using BOTH image and clinical text.
    """
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Text preprocessing
    encoding = tokenizer(
        clinical_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(
            images=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_embeddings=True,
        )
    
    probs = output["probs"][0].cpu().numpy()
    predicted_idx = probs.argmax()
    confidence = probs[predicted_idx]
    
    return {
        "predicted_syndrome": config.syndrome_names[predicted_idx],
        "confidence": float(confidence),
        "all_probabilities": {
            config.syndrome_names[i]: float(probs[i])
            for i in range(len(config.syndrome_names))
        },
        "attention_weights": output.get("attention_info", {})
    }


def main():
    print("=" * 70)
    print("MULTIMODAL RARE DISEASE PREDICTION DEMO")
    print("Using Facial Images + Clinical Text Fusion")
    print("=" * 70)
    
    # Load model
    print("\nLoading multimodal model...")
    model, config, device = load_model()
    print(f"Model loaded on {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder.model_name)
    
    # Load clinical descriptions
    with open("data/syndrome_clinical_descriptions.json", "r") as f:
        clinical_descriptions = json.load(f)
    
    # Find test images
    image_dir = Path("data/images_augmented")
    if not image_dir.exists():
        image_dir = Path("data/images_organized")
    
    # Test on random samples from each syndrome
    print("\n" + "=" * 70)
    print("TESTING PREDICTIONS ON SAMPLE IMAGES")
    print("=" * 70)
    
    correct = 0
    total = 0
    
    for syndrome_folder in sorted(image_dir.iterdir()):
        if not syndrome_folder.is_dir():
            continue
        
        # Map folder name to syndrome
        folder_name = syndrome_folder.name
        syndrome_mapping = {
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
        }
        
        true_syndrome = syndrome_mapping.get(folder_name, folder_name.replace("_", " "))
        
        if true_syndrome not in clinical_descriptions:
            continue
        
        # Get random image
        images = list(syndrome_folder.glob("*.jpg")) + list(syndrome_folder.glob("*.png"))
        if not images:
            continue
        
        test_image = random.choice(images)
        clinical_text = clinical_descriptions[true_syndrome]["clinical_description"]
        
        # Predict
        result = predict_multimodal(model, config, device, str(test_image), clinical_text, tokenizer)
        
        is_correct = result["predicted_syndrome"] == true_syndrome
        correct += int(is_correct)
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} True: {true_syndrome}")
        print(f"  Predicted: {result['predicted_syndrome']} ({result['confidence']*100:.1f}%)")
        print(f"  Image: {test_image.name}")
    
    print("\n" + "=" * 70)
    print(f"DEMO ACCURACY: {correct}/{total} = {100*correct/total:.1f}%")
    print("=" * 70)
    
    # Interactive demo
    print("\n" + "-" * 70)
    print("MULTIMODAL FUSION DEMONSTRATION")
    print("-" * 70)
    
    # Pick one syndrome for detailed demo
    demo_syndrome = "Williams-Beuren Syndrome"
    demo_folder = image_dir / "SYN_WBS"
    if not demo_folder.exists():
        demo_folder = image_dir / "Williams_Beuren_Syndrome"
    
    if demo_folder.exists():
        demo_images = list(demo_folder.glob("*.jpg")) + list(demo_folder.glob("*.png"))
        if demo_images:
            demo_image = demo_images[0]
            demo_text = clinical_descriptions[demo_syndrome]["clinical_description"]
            
            print(f"\nImage: {demo_image.name}")
            print(f"\nClinical Text (truncated):")
            print(f"  '{demo_text[:150]}...'")
            
            result = predict_multimodal(model, config, device, str(demo_image), demo_text, tokenizer)
            
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"  Top Prediction: {result['predicted_syndrome']}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            
            print(f"\n  All Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: -x[1])
            for syndrome, prob in sorted_probs[:5]:
                bar = "█" * int(prob * 30)
                print(f"    {syndrome:35s} {prob*100:5.1f}% {bar}")


if __name__ == "__main__":
    main()
