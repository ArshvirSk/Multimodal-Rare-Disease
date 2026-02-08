"""
Configuration module for Multimodal Rare Disease Diagnosis Framework.
Contains all hyperparameters, paths, and model configurations.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import torch


# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


@dataclass
class DataConfig:
    """Data-related configuration."""

    # Image settings
    image_size: int = 224
    image_channels: int = 3

    # Text settings
    max_text_length: int = 128
    text_model_name: str = "dmis-lab/biobert-base-cased-v1.2"

    # Data paths
    orphadata_diseases: Path = DATA_DIR / "orphadata" / "orphadata_diseases.xml"
    orphadata_phenotypes: Path = DATA_DIR / "orphadata" / "orphadata_phenotypes.xml"
    orphadata_genes: Path = DATA_DIR / "orphadata" / "orphadata_genes.xml"
    hpo_ontology: Path = DATA_DIR / "hpo" / "hp.obo"
    hpo_annotations: Path = DATA_DIR / "hpo" / "phenotype.hpoa"
    fgdd_dir: Path = DATA_DIR / "FGDD"
    pdidb_dir: Path = BASE_DIR / "PDIDB"

    # Dataset splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Data augmentation
    augment_images: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 15
    brightness_factor: float = 0.2

    # Class balance
    use_weighted_sampling: bool = True
    oversample_minority: bool = True


@dataclass
class CNNEncoderConfig:
    """CNN encoder configuration."""

    backbone: str = "resnet50"  # Options: resnet50, efficientnet_b0
    pretrained: bool = True
    embedding_dim: int = 512
    freeze_backbone: bool = True  # IMPORTANT for small data - leverage pretrained weights
    freeze_layers: int = 6  # Freeze first 6 layers of ResNet50 for transfer learning
    dropout: float = 0.5  # Higher dropout for small datasets


@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""

    model_name: str = "dmis-lab/biobert-base-cased-v1.2"
    embedding_dim: int = 768
    max_length: int = 128
    freeze_embeddings: bool = False
    freeze_layers: int = 0  # Number of transformer layers to freeze
    dropout: float = 0.1
    use_pooler_output: bool = False  # Use CLS token if False


@dataclass
class FusionConfig:
    """Multimodal fusion configuration."""

    fusion_type: str = "attention"  # Options: concatenation, attention, gated
    hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout: float = 0.3
    use_residual: bool = True

    # Projection dimensions (must match encoder output dims)
    image_proj_dim: int = 512  # CNN encoder output
    text_proj_dim: int = 768  # BioBERT output dimension


@dataclass
class ClassifierConfig:
    """Classification head configuration."""

    # Smaller for less overfitting
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    num_classes: int = 10  # Number of syndrome classes
    dropout: float = 0.5  # Higher dropout for small datasets
    activation: str = "relu"  # Options: relu, gelu, leaky_relu


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic training params - optimized for small datasets
    batch_size: int = 4  # Small batch for limited data
    num_epochs: int = 50  # Less epochs, more augmentation
    learning_rate: float = 5e-5  # Lower LR for pretrained models
    weight_decay: float = 0.05  # Higher weight decay for regularization

    # Optimizer settings
    optimizer: str = "adamw"  # Options: adam, adamw, sgd
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    warmup_epochs: int = 5

    # Label smoothing for better generalization
    label_smoothing: float = 0.1

    # Learning rate schedule
    lr_decay_factor: float = 0.1
    lr_decay_epochs: List[int] = field(default_factory=lambda: [30, 60, 90])

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.001

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: Path = CHECKPOINTS_DIR

    # Mixed precision training
    use_amp: bool = True

    # Gradient clipping
    gradient_clip_val: float = 1.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "confusion_matrix",
            "roc_auc",
        ]
    )
    per_class_metrics: bool = True
    save_predictions: bool = True
    results_dir: Path = RESULTS_DIR


@dataclass
class ExplainabilityConfig:
    """Explainability configuration."""

    use_gradcam: bool = True
    gradcam_layer: str = "layer4"  # For ResNet
    use_attention_viz: bool = True
    save_visualizations: bool = True
    num_samples_to_visualize: int = 10


@dataclass
class Config:
    """Master configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    cnn_encoder: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = field(
        default_factory=ExplainabilityConfig)

    # Disease/syndrome mapping
    syndrome_names: List[str] = field(
        default_factory=lambda: [
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
    )

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.evaluation.results_dir, exist_ok=True)


# Default configuration instance
config = Config()


def get_config() -> Config:
    """Get the default configuration."""
    return config


def update_config(**kwargs) -> Config:
    """Update configuration with custom values."""
    # This is a simplified update - in production, use a proper config library
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
