"""
CNN Encoder for Facial Image Feature Extraction.
Supports ResNet50 and EfficientNet-B0 backbones with pretrained weights.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

from .config import get_config, CNNEncoderConfig


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting facial image embeddings.

    Supports:
    - ResNet50 (default)
    - EfficientNet-B0

    Output: 512-dimensional embedding vector
    """

    def __init__(self, config: Optional[CNNEncoderConfig] = None):
        """
        Initialize CNN encoder.

        Args:
            config: CNN encoder configuration. Uses default if None.
        """
        super().__init__()

        if config is None:
            config = get_config().cnn_encoder

        self.config = config
        self.backbone_name = config.backbone
        self.embedding_dim = config.embedding_dim

        # Build backbone
        self.backbone, backbone_out_features = self._build_backbone()

        # Projection head to get desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_features, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        # Optionally freeze backbone layers
        if config.freeze_backbone:
            self._freeze_backbone()
        elif config.freeze_layers > 0:
            self._freeze_layers(config.freeze_layers)

    def _build_backbone(self) -> Tuple[nn.Module, int]:
        """
        Build the CNN backbone.

        Returns:
            Tuple of (backbone module, output feature dimension)
        """
        if self.backbone_name == "resnet50":
            return self._build_resnet50()
        elif self.backbone_name == "efficientnet_b0":
            return self._build_efficientnet_b0()
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

    def _build_resnet50(self) -> Tuple[nn.Module, int]:
        """Build ResNet50 backbone."""
        if self.config.pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
        else:
            backbone = models.resnet50(weights=None)

        # Remove the final classification layer
        # ResNet50 outputs 2048-d features before fc
        out_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, out_features

    def _build_efficientnet_b0(self) -> Tuple[nn.Module, int]:
        """Build EfficientNet-B0 backbone."""
        if self.config.pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = models.efficientnet_b0(weights=weights)
        else:
            backbone = models.efficientnet_b0(weights=None)

        # EfficientNet-B0 outputs 1280-d features before classifier
        out_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        return backbone, out_features

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Froze all backbone parameters")

    def _freeze_layers(self, num_layers: int):
        """
        Freeze the first N layers of the backbone.
        For small datasets, freeze most layers to leverage pretrained features.

        Args:
            num_layers: Number of layers to freeze (0-4 for ResNet)
                       Higher = more frozen = better for very small data
        """
        if self.backbone_name == "resnet50":
            # ResNet has: conv1, bn1, relu, maxpool, layer1-4
            # Effective layers for freezing: conv1+bn1, layer1, layer2, layer3, layer4
            layers_to_freeze = ["conv1", "bn1", "relu", "maxpool"]

            if num_layers >= 1:
                layers_to_freeze.append("layer1")
            if num_layers >= 2:
                layers_to_freeze.append("layer2")
            if num_layers >= 3:
                layers_to_freeze.append("layer3")
            if num_layers >= 4:
                # Only fine-tune projection head
                layers_to_freeze.append("layer4")

            # Count frozen parameters for logging
            frozen_params = 0
            total_params = 0

            for name, module in self.backbone.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                total_params += module_params

                if name in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen_params += module_params

            print(f"Froze layers: {layers_to_freeze}")
            print(
                f"Frozen backbone params: {frozen_params:,} / {total_params:,} ({100*frozen_params/total_params:.1f}%)")

        elif self.backbone_name == "efficientnet_b0":
            # Freeze features blocks
            frozen = 0
            frozen_params = 0
            total_params = sum(p.numel()
                               for p in self.backbone.features.parameters())

            for idx, block in enumerate(self.backbone.features):
                if frozen >= num_layers:
                    break
                for param in block.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                frozen += 1

            print(f"Froze {frozen} EfficientNet blocks")
            print(
                f"Frozen backbone params: {frozen_params:,} / {total_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor [batch_size, 3, 224, 224]

        Returns:
            Embedding tensor [batch_size, embedding_dim]
        """
        # Extract backbone features
        features = self.backbone(x)  # [batch_size, backbone_features]

        # Project to embedding dimension
        embedding = self.projection(features)  # [batch_size, embedding_dim]

        return embedding

    def get_attention_layer(self) -> nn.Module:
        """
        Get the attention layer for Grad-CAM visualization.

        Returns:
            The layer to use for Grad-CAM
        """
        if self.backbone_name == "resnet50":
            return self.backbone.layer4
        elif self.backbone_name == "efficientnet_b0":
            return self.backbone.features[-1]
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

    def get_intermediate_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get intermediate features for visualization.

        Args:
            x: Input image tensor

        Returns:
            Tuple of (feature_maps, embedding)
        """
        if self.backbone_name == "resnet50":
            # Get features before global average pooling
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            feature_maps = self.backbone.layer4(x)

            # Global average pooling
            features = self.backbone.avgpool(feature_maps)
            features = torch.flatten(features, 1)

            # Projection
            embedding = self.projection(features)

            return feature_maps, embedding

        elif self.backbone_name == "efficientnet_b0":
            feature_maps = self.backbone.features(x)
            features = self.backbone.avgpool(feature_maps)
            features = torch.flatten(features, 1)
            embedding = self.projection(features)

            return feature_maps, embedding

        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")


class ResNet50Encoder(CNNEncoder):
    """Convenience class for ResNet50 encoder."""

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_layers: int = 0,
    ):
        config = CNNEncoderConfig(
            backbone="resnet50",
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )
        super().__init__(config)


class EfficientNetEncoder(CNNEncoder):
    """Convenience class for EfficientNet-B0 encoder."""

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_layers: int = 0,
    ):
        config = CNNEncoderConfig(
            backbone="efficientnet_b0",
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )
        super().__init__(config)


def create_cnn_encoder(
    backbone: str = "resnet50",
    embedding_dim: int = 512,
    pretrained: bool = True,
    **kwargs,
) -> CNNEncoder:
    """
    Factory function to create CNN encoder.

    Args:
        backbone: Backbone architecture name
        embedding_dim: Output embedding dimension
        pretrained: Use pretrained weights
        **kwargs: Additional configuration options

    Returns:
        Initialized CNN encoder
    """
    config = CNNEncoderConfig(
        backbone=backbone, embedding_dim=embedding_dim, pretrained=pretrained, **kwargs
    )
    return CNNEncoder(config)


if __name__ == "__main__":
    # Test the CNN encoder
    print("Testing CNN Encoder...")

    # Test ResNet50
    print("\n--- ResNet50 Encoder ---")
    resnet_encoder = ResNet50Encoder(embedding_dim=512)
    print(f"Backbone: ResNet50")
    print(f"Output dim: {resnet_encoder.embedding_dim}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = resnet_encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test EfficientNet
    print("\n--- EfficientNet-B0 Encoder ---")
    efficientnet_encoder = EfficientNetEncoder(embedding_dim=512)
    print(f"Backbone: EfficientNet-B0")
    print(f"Output dim: {efficientnet_encoder.embedding_dim}")

    output = efficientnet_encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    resnet_params = sum(p.numel() for p in resnet_encoder.parameters())
    efficientnet_params = sum(p.numel()
                              for p in efficientnet_encoder.parameters())
    print(f"\nResNet50 parameters: {resnet_params:,}")
    print(f"EfficientNet-B0 parameters: {efficientnet_params:,}")
