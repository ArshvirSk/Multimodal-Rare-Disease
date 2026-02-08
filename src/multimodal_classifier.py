"""
Multimodal Classifier.
Combines CNN encoder, text encoder, and fusion module for syndrome classification.
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn

from .config import get_config, Config
from .cnn_encoder import CNNEncoder, create_cnn_encoder
from .text_encoder import TextEncoder, create_text_encoder
from .fusion_model import MultimodalFusion, create_fusion_module


class ClassificationHead(nn.Module):
    """
    Classification head for syndrome prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256],
        num_classes: int = 10,
        dropout: float = 0.4,
        activation: str = "relu",
    ):
        """
        Initialize classification head.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
            activation: Activation function (relu, gelu, leaky_relu)
        """
        super().__init__()

        self.num_classes = num_classes

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "gelu":
            return nn.GELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.1, inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits [batch_size, num_classes]
        """
        return self.classifier(x)


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal classifier for rare disease diagnosis.

    Combines:
    - CNN encoder for facial images
    - Transformer encoder for clinical text
    - Multimodal fusion module
    - Classification head
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize multimodal classifier.

        Args:
            config: Configuration object. Uses default if None.
        """
        super().__init__()

        if config is None:
            config = get_config()

        self.config = config

        # Build components
        self.cnn_encoder = CNNEncoder(config.cnn_encoder)
        self.text_encoder = TextEncoder(config.text_encoder)
        self.fusion = MultimodalFusion(config.fusion)

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=config.fusion.hidden_dim,
            hidden_dims=config.classifier.hidden_dims,
            num_classes=config.classifier.num_classes,
            dropout=config.classifier.dropout,
            activation=config.classifier.activation,
        )

        # Store dimensions for reference
        self.image_embedding_dim = config.cnn_encoder.embedding_dim
        self.text_embedding_dim = config.text_encoder.embedding_dim
        self.fusion_dim = config.fusion.hidden_dim
        self.num_classes = config.classifier.num_classes

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Image tensor [batch_size, 3, 224, 224]
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with:
            - logits: Classification logits
            - probs: Classification probabilities
            - image_embedding: (optional) Image embedding
            - text_embedding: (optional) Text embedding
            - fused_embedding: (optional) Fused embedding
            - attention_info: (optional) Attention weights
        """
        # Encode image
        image_embedding = self.cnn_encoder(images)

        # Encode text
        text_embedding = self.text_encoder(input_ids, attention_mask)

        # Fuse modalities
        fused_embedding, attention_info = self.fusion(image_embedding, text_embedding)

        # Classify
        logits = self.classifier(fused_embedding)
        probs = torch.softmax(logits, dim=-1)

        output = {"logits": logits, "probs": probs}

        if return_embeddings:
            output["image_embedding"] = image_embedding
            output["text_embedding"] = text_embedding
            output["fused_embedding"] = fused_embedding
            output["attention_info"] = attention_info

        return output

    def predict(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.

        Args:
            images: Image tensor
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(images, input_ids, attention_mask)
            probs = output["probs"]
            confidence, predicted = torch.max(probs, dim=-1)

        return predicted, confidence


class ImageOnlyClassifier(nn.Module):
    """
    Image-only baseline classifier (unimodal).
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize image-only classifier.

        Args:
            config: Configuration object
        """
        super().__init__()

        if config is None:
            config = get_config()

        self.cnn_encoder = CNNEncoder(config.cnn_encoder)

        self.classifier = ClassificationHead(
            input_dim=config.cnn_encoder.embedding_dim,
            hidden_dims=config.classifier.hidden_dims,
            num_classes=config.classifier.num_classes,
            dropout=config.classifier.dropout,
            activation=config.classifier.activation,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Image tensor [batch_size, 3, 224, 224]

        Returns:
            Dictionary with logits and probs
        """
        embedding = self.cnn_encoder(images)
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=-1)

        return {"logits": logits, "probs": probs}


class TextOnlyClassifier(nn.Module):
    """
    Text-only baseline classifier (unimodal).
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize text-only classifier.

        Args:
            config: Configuration object
        """
        super().__init__()

        if config is None:
            config = get_config()

        self.text_encoder = TextEncoder(config.text_encoder)

        self.classifier = ClassificationHead(
            input_dim=config.text_encoder.embedding_dim,
            hidden_dims=config.classifier.hidden_dims,
            num_classes=config.classifier.num_classes,
            dropout=config.classifier.dropout,
            activation=config.classifier.activation,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Dictionary with logits and probs
        """
        embedding = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=-1)

        return {"logits": logits, "probs": probs}


def create_multimodal_classifier(
    num_classes: int = 10,
    cnn_backbone: str = "resnet50",
    text_model: str = "dmis-lab/biobert-base-cased-v1.2",
    fusion_type: str = "attention",
    **kwargs,
) -> MultimodalClassifier:
    """
    Factory function to create multimodal classifier.

    Args:
        num_classes: Number of syndrome classes
        cnn_backbone: CNN backbone architecture
        text_model: Text encoder model name
        fusion_type: Fusion type (concatenation, attention, gated)
        **kwargs: Additional configuration options

    Returns:
        Initialized multimodal classifier
    """
    config = get_config()

    # Update configuration
    config.classifier.num_classes = num_classes
    config.cnn_encoder.backbone = cnn_backbone
    config.text_encoder.model_name = text_model
    config.fusion.fusion_type = fusion_type

    return MultimodalClassifier(config)


def create_baseline_classifiers(
    config: Optional[Config] = None,
) -> Tuple[ImageOnlyClassifier, TextOnlyClassifier]:
    """
    Create baseline unimodal classifiers.

    Args:
        config: Configuration object

    Returns:
        Tuple of (image_classifier, text_classifier)
    """
    return ImageOnlyClassifier(config), TextOnlyClassifier(config)


if __name__ == "__main__":
    # Test the classifiers
    print("Testing Multimodal Classifier...")

    # Configuration
    config = get_config()
    config.classifier.num_classes = 10

    batch_size = 2
    seq_length = 128

    # Create dummy inputs
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    print(f"Image shape: {images.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Test multimodal classifier
    print("\n--- Multimodal Classifier ---")
    try:
        multimodal_clf = MultimodalClassifier(config)
        output = multimodal_clf(
            images, input_ids, attention_mask, return_embeddings=True
        )

        print(f"Logits shape: {output['logits'].shape}")
        print(f"Probs shape: {output['probs'].shape}")
        print(f"Image embedding shape: {output['image_embedding'].shape}")
        print(f"Text embedding shape: {output['text_embedding'].shape}")
        print(f"Fused embedding shape: {output['fused_embedding'].shape}")

        # Test prediction
        predicted, confidence = multimodal_clf.predict(
            images, input_ids, attention_mask
        )
        print(f"Predictions: {predicted}")
        print(f"Confidence: {confidence}")

        # Count parameters
        total_params = sum(p.numel() for p in multimodal_clf.parameters())
        trainable_params = sum(
            p.numel() for p in multimodal_clf.parameters() if p.requires_grad
        )
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: You may need to download the pretrained models first")

    # Test image-only baseline
    print("\n--- Image-Only Baseline ---")
    image_clf = ImageOnlyClassifier(config)
    output = image_clf(images)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Image-only params: {sum(p.numel() for p in image_clf.parameters()):,}")

    # Test text-only baseline
    print("\n--- Text-Only Baseline ---")
    try:
        text_clf = TextOnlyClassifier(config)
        output = text_clf(input_ids, attention_mask)
        print(f"Logits shape: {output['logits'].shape}")
        print(f"Text-only params: {sum(p.numel() for p in text_clf.parameters()):,}")
    except Exception as e:
        print(f"Error: {e}")
