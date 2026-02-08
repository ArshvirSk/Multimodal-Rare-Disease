"""
Inference Pipeline for Multimodal Rare Disease Diagnosis.
End-to-end prediction from image and clinical text input.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms

from .config import get_config, Config
from .multimodal_classifier import MultimodalClassifier


class MultimodalPredictor:
    """
    End-to-end predictor for rare disease diagnosis.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration object
            device: Device to run inference on
        """
        if config is None:
            config = get_config()

        self.config = config
        self.device = torch.device(device or config.training.device)

        # Load model
        self.model = MultimodalClassifier(config)

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        # Image transforms
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder.model_name)
        self.max_length = config.text_encoder.max_length

        # Class names
        self.class_names = config.syndrome_names

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: Image path or PIL Image

        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Invalid image type: {type(image)}")

        return self.image_transform(image)

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess clinical text for inference.

        Args:
            text: Clinical narrative text

        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        text: str,
        top_k: int = 3,
        return_embeddings: bool = False,
    ) -> Dict:
        """
        Make prediction for a single sample.

        Args:
            image: Facial image
            text: Clinical narrative text
            top_k: Number of top predictions to return
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        image_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        text_encoding = self.preprocess_text(text)
        input_ids = text_encoding["input_ids"].to(self.device)
        attention_mask = text_encoding["attention_mask"].to(self.device)

        # Forward pass
        output = self.model(
            images=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_embeddings=return_embeddings,
        )

        # Get predictions
        probs = output["probs"][0].cpu().numpy()

        # Get top-k predictions
        top_indices = probs.argsort()[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append(
                {
                    "syndrome": (
                        self.class_names[idx]
                        if idx < len(self.class_names)
                        else f"Class_{idx}"
                    ),
                    "class_id": int(idx),
                    "confidence": float(probs[idx]),
                    "probability_percent": float(probs[idx] * 100),
                }
            )

        result = {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "all_probabilities": {
                (
                    self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                ): float(probs[i])
                for i in range(len(probs))
            },
        }

        if return_embeddings:
            result["embeddings"] = {
                "image": output["image_embedding"][0].cpu().numpy().tolist(),
                "text": output["text_embedding"][0].cpu().numpy().tolist(),
                "fused": output["fused_embedding"][0].cpu().numpy().tolist(),
            }

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        texts: List[str],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Make predictions for a batch of samples.

        Args:
            images: List of facial images
            texts: List of clinical narratives
            top_k: Number of top predictions per sample

        Returns:
            List of prediction results
        """
        assert len(images) == len(texts), "Number of images must match number of texts"

        # Preprocess images
        image_tensors = torch.stack([self.preprocess_image(img) for img in images]).to(
            self.device
        )

        # Preprocess texts
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Forward pass
        output = self.model(
            images=image_tensors, input_ids=input_ids, attention_mask=attention_mask
        )

        # Process each sample
        probs = output["probs"].cpu().numpy()
        results = []

        for i, sample_probs in enumerate(probs):
            top_indices = sample_probs.argsort()[::-1][:top_k]

            predictions = []
            for idx in top_indices:
                predictions.append(
                    {
                        "syndrome": (
                            self.class_names[idx]
                            if idx < len(self.class_names)
                            else f"Class_{idx}"
                        ),
                        "class_id": int(idx),
                        "confidence": float(sample_probs[idx]),
                    }
                )

            results.append(
                {
                    "sample_idx": i,
                    "predictions": predictions,
                    "top_prediction": predictions[0] if predictions else None,
                }
            )

        return results

    def format_report(self, prediction: Dict) -> str:
        """
        Format prediction as a clinical report.

        Args:
            prediction: Prediction dictionary

        Returns:
            Formatted report string
        """
        top = prediction["top_prediction"]

        report = []
        report.append("=" * 60)
        report.append("MULTIMODAL RARE DISEASE DIAGNOSIS REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("PRIMARY DIAGNOSIS:")
        report.append(f"  Syndrome: {top['syndrome']}")
        report.append(f"  Confidence: {top['probability_percent']:.1f}%")
        report.append("")
        report.append("DIFFERENTIAL DIAGNOSES:")

        for i, pred in enumerate(prediction["predictions"][1:], start=2):
            report.append(
                f"  {i}. {pred['syndrome']} ({pred['probability_percent']:.1f}%)"
            )

        report.append("")
        report.append("-" * 60)
        report.append("NOTE: This is an AI-assisted diagnosis tool.")
        report.append("Final diagnosis should be confirmed by a specialist.")
        report.append("=" * 60)

        return "\n".join(report)


def predict_from_files(
    image_path: str,
    text_path: Optional[str] = None,
    text: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Make prediction from file paths.

    Args:
        image_path: Path to facial image
        text_path: Path to text file with clinical narrative
        text: Direct clinical narrative text (alternative to text_path)
        checkpoint_path: Path to model checkpoint
        output_path: Path to save prediction JSON

    Returns:
        Prediction dictionary
    """
    config = get_config()

    # Load text
    if text is None and text_path:
        with open(text_path, "r") as f:
            text = f.read()
    elif text is None:
        text = "No clinical narrative provided."

    # Initialize predictor
    predictor = MultimodalPredictor(
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        config=config,
    )

    # Make prediction
    result = predictor.predict(
        image=image_path, text=text, top_k=5, return_embeddings=False
    )

    # Print report
    print(predictor.format_report(result))

    # Save if requested
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved prediction to {output_path}")

    return result


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Predict syndrome from image and text")
    parser.add_argument("--image", type=str, required=True, help="Path to facial image")
    parser.add_argument("--text", type=str, help="Clinical narrative text")
    parser.add_argument("--text_file", type=str, help="Path to clinical narrative file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Output path for prediction JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample data")

    args = parser.parse_args()

    if args.demo:
        print("Running inference demo...")
        print("\nDemo requires a trained model checkpoint.")
        print("To train a model first, run:")
        print("  python -m src.train --mode multimodal --smoke_test")
        print("\nThen run inference:")
        print(
            "  python -m src.predict --image path/to/image.jpg --text 'Patient presents with...' --checkpoint checkpoints/multimodal_best.pt"
        )
        return

    # Get text
    text = args.text
    if not text and args.text_file:
        with open(args.text_file, "r") as f:
            text = f.read()

    if not text:
        text = "Patient presents with characteristic facial features."
        print(f"Warning: No clinical text provided. Using default: '{text}'")

    # Run prediction
    try:
        result = predict_from_files(
            image_path=args.image,
            text=text,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("\nMake sure you have:")
        print("1. A valid image path")
        print("2. A trained model checkpoint")
        print("3. Required dependencies installed")


if __name__ == "__main__":
    main()
