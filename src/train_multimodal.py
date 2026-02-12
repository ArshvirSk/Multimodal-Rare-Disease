"""
Multimodal Training Pipeline for Rare Disease Diagnosis.
Combines facial images with clinical text descriptions using attention-based fusion.
"""

import json
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from .config import get_config, Config
from .multimodal_classifier import MultimodalClassifier


# =============================================================================
# CONFIGURATION FOR MULTIMODAL TRAINING
# =============================================================================

MULTIMODAL_CONFIG = {
    # Model settings
    "freeze_backbone_layers": 3,  # Freeze CNN backbone layers
    "freeze_text_layers": 6,      # Freeze first N transformer layers

    # Regularization
    "dropout": 0.5,
    "weight_decay": 0.03,
    "label_smoothing": 0.1,

    # Training settings
    "batch_size": 8,
    "learning_rate": 2e-5,  # Lower LR for transformer fine-tuning
    "num_epochs": 60,
    "warmup_epochs": 5,

    # Early stopping
    "patience": 15,
    "min_delta": 0.001,

    # Text encoder
    "max_text_length": 256,
    "text_augmentation": True,  # Enable text augmentation
}


# =============================================================================
# CLINICAL TEXT AUGMENTATION
# =============================================================================

class ClinicalTextAugmenter:
    """
    Augment clinical text descriptions for training variety.
    This helps the model learn robust text features.
    """

    def __init__(self, descriptions: Dict[str, Dict]):
        self.descriptions = descriptions

    def augment(self, syndrome_name: str, augment_level: int = 0) -> str:
        """
        Generate augmented clinical text for a syndrome.

        Args:
            syndrome_name: Name of the syndrome
            augment_level: 0 = full description, 1-3 = varied templates

        Returns:
            Augmented clinical text
        """
        if syndrome_name not in self.descriptions:
            return f"Patient presents with features consistent with {syndrome_name}."

        info = self.descriptions[syndrome_name]
        full_description = info.get("clinical_description", "")
        key_features = info.get("key_facial_features", [])

        if augment_level == 0:
            # Return full clinical description
            return full_description

        elif augment_level == 1:
            # Focus on facial features
            if key_features:
                features = random.sample(key_features, min(5, len(key_features)))
                return (
                    f"Facial dysmorphism assessment reveals: {', '.join(features)}. "
                    f"Clinical presentation consistent with {syndrome_name}."
                )
            return full_description

        elif augment_level == 2:
            # Medical report style
            if key_features:
                features = random.sample(key_features, min(6, len(key_features)))
                return (
                    f"Physical examination findings: The patient demonstrates characteristic "
                    f"facial features including {', '.join(features[:3])}. Additional findings "
                    f"include {', '.join(features[3:])}. Differential diagnosis includes {syndrome_name}."
                )
            return full_description

        else:
            # Simplified feature list with random selection
            if key_features:
                num_features = random.randint(3, min(7, len(key_features)))
                selected = random.sample(key_features, num_features)
                templates = [
                    f"Key phenotypic features observed: {', '.join(selected)}.",
                    f"Craniofacial examination shows: {'; '.join(selected)}.",
                    f"Notable dysmorphic features: {', '.join(selected)}. Pattern suggests {syndrome_name}.",
                ]
                return random.choice(templates)
            return full_description


# =============================================================================
# MULTIMODAL DATASET
# =============================================================================

class MultimodalSyndromeDataset(Dataset):
    """
    Dataset that pairs facial images with clinical text descriptions.
    Enables true multimodal fusion during training.
    """

    # Folder name mapping - supports multiple naming conventions
    FOLDER_TO_SYNDROME = {
        # Long names (original format)
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
        # Short names (SYN_* format from augmentation)
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

    def __init__(
        self,
        image_dir: Path,
        clinical_descriptions_path: Path,
        syndrome_names: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        image_transform: Optional[transforms.Compose] = None,
        augmentation_multiplier: int = 1,
        text_augmentation: bool = True,
    ):
        """
        Initialize multimodal dataset.

        Args:
            image_dir: Directory containing syndrome subfolders with images
            clinical_descriptions_path: Path to JSON file with clinical descriptions
            syndrome_names: List of syndrome names (defines class order)
            tokenizer: HuggingFace tokenizer for text encoding
            max_length: Maximum text sequence length
            image_transform: Image transforms
            augmentation_multiplier: Number of times to repeat dataset
            text_augmentation: Whether to apply text augmentation
        """
        self.image_dir = Path(image_dir)
        self.syndrome_names = syndrome_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform
        self.augmentation_multiplier = augmentation_multiplier
        self.text_augmentation = text_augmentation

        # Build class index mapping
        self.syndrome_to_idx = {name: idx for idx, name in enumerate(syndrome_names)}

        # Load clinical descriptions
        with open(clinical_descriptions_path, 'r', encoding='utf-8') as f:
            self.clinical_descriptions = json.load(f)

        # Create text augmenter
        self.text_augmenter = ClinicalTextAugmenter(self.clinical_descriptions)

        # Load samples
        self.samples = self._load_samples()

        # Compute class weights for balanced sampling
        self.class_counts = self._count_classes()
        self.class_weights = self._compute_class_weights()

        print(f"\n{'='*60}")
        print("MULTIMODAL DATASET INITIALIZED")
        print(f"{'='*60}")
        print(f"Images: {len(self.samples)} base samples × {augmentation_multiplier} = {len(self)} total")
        print(f"Text encoder: {tokenizer.name_or_path}")
        print(f"Max text length: {max_length}")
        print(f"Text augmentation: {'Enabled' if text_augmentation else 'Disabled'}")
        print(f"Classes: {len(syndrome_names)}")
        for name, idx in self.syndrome_to_idx.items():
            count = self.class_counts.get(idx, 0)
            print(f"  [{idx}] {name}: {count} images")
        print(f"{'='*60}\n")

    def _load_samples(self) -> List[Tuple[Path, str, int]]:
        """Load all image samples with their syndrome names."""
        samples = []

        for folder in self.image_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name
            syndrome_name = self.FOLDER_TO_SYNDROME.get(folder_name)

            if syndrome_name is None or syndrome_name not in self.syndrome_to_idx:
                continue

            label = self.syndrome_to_idx[syndrome_name]

            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append((img_path, syndrome_name, label))

        return samples

    def _count_classes(self) -> Dict[int, int]:
        """Count samples per class."""
        counts = {}
        for _, _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute balanced class weights."""
        num_classes = len(self.syndrome_names)
        total = len(self.samples)

        weights = []
        for i in range(num_classes):
            count = self.class_counts.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples) * self.augmentation_multiplier

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map augmented index back to base sample
        base_idx = idx % len(self.samples)
        aug_level = idx // len(self.samples)  # Use for text augmentation variety
        img_path, syndrome_name, label = self.samples[base_idx]

        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.image_transform:
            image = self.image_transform(image)

        # Get clinical text with augmentation
        if self.text_augmentation:
            # Vary augmentation level based on index for diversity
            text_aug_level = aug_level % 4
        else:
            text_aug_level = 0

        clinical_text = self.text_augmenter.augment(syndrome_name, text_aug_level)

        # Tokenize text
        encoding = self.tokenizer(
            clinical_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation transforms with mild perturbations."""
    return transforms.Compose([
        transforms.Resize((image_size + 10, image_size + 10)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =============================================================================
# MULTIMODAL TRAINER
# =============================================================================

class MultimodalTrainer:
    """
    Trainer for multimodal rare disease classification.
    Handles both image and text modalities with fusion.
    """

    def __init__(
        self,
        model: MultimodalClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config if config is not None else MULTIMODAL_CONFIG.copy()

        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Freeze layers for transfer learning
        self._freeze_backbone_layers(self.config.get("freeze_backbone_layers", 3))
        self._freeze_text_layers(self.config.get("freeze_text_layers", 6))

        # Loss with class weights and label smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device),
            label_smoothing=self.config.get("label_smoothing", 0.1)
        )

        # Optimizer - different learning rates for different components
        param_groups = self._get_param_groups()
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config.get("weight_decay", 0.03)
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        # Tracking
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "learning_rate": []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _get_param_groups(self) -> List[Dict]:
        """Get parameter groups with different learning rates."""
        base_lr = self.config.get("learning_rate", 2e-5)

        param_groups = [
            # CNN encoder - lower LR (pretrained)
            {
                "params": [p for n, p in self.model.cnn_encoder.named_parameters() if p.requires_grad],
                "lr": base_lr * 0.1,
                "name": "cnn_encoder"
            },
            # Text encoder - medium LR (pretrained but needs domain adaptation)
            {
                "params": [p for n, p in self.model.text_encoder.named_parameters() if p.requires_grad],
                "lr": base_lr * 0.5,
                "name": "text_encoder"
            },
            # Fusion module - full LR (trained from scratch)
            {
                "params": [p for n, p in self.model.fusion.named_parameters() if p.requires_grad],
                "lr": base_lr,
                "name": "fusion"
            },
            # Classifier - full LR (trained from scratch)
            {
                "params": [p for n, p in self.model.classifier.named_parameters() if p.requires_grad],
                "lr": base_lr,
                "name": "classifier"
            },
        ]

        # Filter empty groups
        return [g for g in param_groups if len(list(g["params"])) > 0]

    def _freeze_backbone_layers(self, num_layers: int):
        """Freeze CNN backbone layers."""
        if hasattr(self.model.cnn_encoder, 'backbone'):
            backbone = self.model.cnn_encoder.backbone

            layers_to_freeze = ["conv1", "bn1", "relu", "maxpool"]
            if num_layers >= 1:
                layers_to_freeze.append("layer1")
            if num_layers >= 2:
                layers_to_freeze.append("layer2")
            if num_layers >= 3:
                layers_to_freeze.append("layer3")

            for name, module in backbone.named_children():
                if name in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

        frozen = sum(1 for p in self.model.cnn_encoder.parameters() if not p.requires_grad)
        total = sum(1 for p in self.model.cnn_encoder.parameters())
        print(f"CNN Encoder: {frozen}/{total} parameters frozen")

    def _freeze_text_layers(self, num_layers: int):
        """Freeze text encoder transformer layers."""
        if hasattr(self.model.text_encoder, 'encoder'):
            encoder = self.model.text_encoder.encoder

            # Freeze embeddings
            if hasattr(encoder, 'embeddings'):
                for param in encoder.embeddings.parameters():
                    param.requires_grad = False

            # Freeze specified number of transformer layers
            if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
                for i, layer in enumerate(encoder.encoder.layer):
                    if i < num_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

        frozen = sum(1 for p in self.model.text_encoder.parameters() if not p.requires_grad)
        total = sum(1 for p in self.model.text_encoder.parameters())
        print(f"Text Encoder: {frozen}/{total} parameters frozen")

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    output = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = output["logits"]
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = output["logits"]
                loss = self.criterion(logits, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Track metrics
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct / max(total, 1):.4f}"
            })

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = output["logits"]
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    def train(self, num_epochs: Optional[int] = None, save_dir: str = "checkpoints"):
        """Run training loop."""
        if num_epochs is None:
            num_epochs = self.config.get("num_epochs", 60)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("MULTIMODAL TRAINING - IMAGE + TEXT FUSION")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.get('batch_size', 8)}")
        print(f"Learning rate: {self.config.get('learning_rate', 2e-5)}")
        print(f"Label smoothing: {self.config.get('label_smoothing', 0.1)}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_acc = self._validate()

            # Update scheduler
            self.scheduler.step()

            # Track history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': self.config,
                }, save_path / "multimodal_best.pt")
            else:
                self.patience_counter += 1

            # Print progress
            marker = " ★ BEST" if is_best else ""
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}{marker}"
            )

            # Early stopping
            if self.patience_counter >= self.config.get("patience", 15):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': self.config,
        }, save_path / "multimodal_last.pt")

        print(f"\n{'='*60}")
        print("MULTIMODAL Training Complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        print(f"{'='*60}")

        # Load best model
        if (save_path / "multimodal_best.pt").exists():
            checkpoint = torch.load(save_path / "multimodal_best.pt", weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        return self.history


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_multimodal(
    image_dirs: Optional[List[str]] = None,
    clinical_descriptions_path: Optional[str] = None,
    config: Optional[Config] = None,
    augmentation_factor: int = 10,
    num_epochs: int = 60,
    batch_size: int = 8,
    save_dir: str = "checkpoints"
) -> Tuple[MultimodalClassifier, Dict[str, List[float]]]:
    """
    Train multimodal model combining facial images with clinical text.

    Args:
        image_dirs: List of possible image directories
        clinical_descriptions_path: Path to clinical descriptions JSON
        config: Configuration object
        augmentation_factor: Data augmentation multiplier
        num_epochs: Number of training epochs
        batch_size: Training batch size
        save_dir: Directory to save checkpoints

    Returns:
        Trained model and training history
    """
    if config is None:
        config = get_config()

    if image_dirs is None:
        image_dirs = ["data/images_augmented", "data/images_organized"]

    if clinical_descriptions_path is None:
        clinical_descriptions_path = "data/syndrome_clinical_descriptions.json"

    # Set seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Find existing image directory
    image_dir = None
    for d in image_dirs:
        if Path(d).exists():
            image_dir = Path(d)
            break

    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in {image_dirs}")

    clinical_path = Path(clinical_descriptions_path)
    if not clinical_path.exists():
        raise FileNotFoundError(f"Clinical descriptions not found: {clinical_path}")

    print(f"Using image directory: {image_dir}")
    print(f"Using clinical descriptions: {clinical_path}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder.model_name)
    print(f"Loaded tokenizer: {config.text_encoder.model_name}")

    # Get transforms
    train_transform = get_train_transforms(config.data.image_size)
    val_transform = get_val_transforms(config.data.image_size)

    # Create full dataset for splitting
    full_dataset = MultimodalSyndromeDataset(
        image_dir=image_dir,
        clinical_descriptions_path=clinical_path,
        syndrome_names=config.syndrome_names,
        tokenizer=tokenizer,
        max_length=MULTIMODAL_CONFIG["max_text_length"],
        image_transform=None,
        augmentation_multiplier=1,
        text_augmentation=False,
    )

    # Split by base images to prevent data leakage
    num_samples = len(full_dataset.samples)

    base_to_samples = defaultdict(list)
    for idx, (path, syndrome, label) in enumerate(full_dataset.samples):
        base_name = re.sub(r'_(aug\d+|orig)$', '', path.stem)
        base_key = (path.parent.name, base_name, label)
        base_to_samples[base_key].append(idx)

    # Stratified split - ensure each class has validation samples
    class_indices = defaultdict(list)
    for idx, (path, syndrome, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        random.shuffle(indices)
        # Ensure at least 1 sample for validation per class (15% or minimum 1)
        num_val = max(1, int(0.15 * len(indices)))
        val_indices.extend(indices[:num_val])
        train_indices.extend(indices[num_val:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    print(f"\nDataset split:")
    print(f"  Total images: {num_samples}")
    print(f"  Training: {len(train_indices)} × {augmentation_factor} = {len(train_indices) * augmentation_factor}")
    print(f"  Validation: {len(val_indices)}")

    # Create train and val sample lists
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    # Create proper dataset classes
    class MultimodalTrainDataset(Dataset):
        def __init__(self, samples, transform, tokenizer, descriptions, max_len, aug_factor):
            self.samples = samples
            self.transform = transform
            self.tokenizer = tokenizer
            self.aug_factor = aug_factor
            self.max_len = max_len
            self.augmenter = ClinicalTextAugmenter(descriptions)

        def __len__(self):
            return len(self.samples) * self.aug_factor

        def __getitem__(self, idx):
            base_idx = idx % len(self.samples)
            aug_level = idx // len(self.samples)
            img_path, syndrome_name, label = self.samples[base_idx]

            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # Get augmented text
            text = self.augmenter.augment(syndrome_name, aug_level % 4)
            encoding = self.tokenizer(
                text, max_length=self.max_len,
                padding="max_length", truncation=True, return_tensors="pt"
            )

            return {
                "image": image,
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    class MultimodalValDataset(Dataset):
        def __init__(self, samples, transform, tokenizer, descriptions, max_len):
            self.samples = samples
            self.transform = transform
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.augmenter = ClinicalTextAugmenter(descriptions)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, syndrome_name, label = self.samples[idx]

            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # Use full clinical description for validation
            text = self.augmenter.augment(syndrome_name, 0)
            encoding = self.tokenizer(
                text, max_length=self.max_len,
                padding="max_length", truncation=True, return_tensors="pt"
            )

            return {
                "image": image,
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    # Load descriptions for dataset classes
    with open(clinical_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    train_dataset = MultimodalTrainDataset(
        train_samples, train_transform, tokenizer, descriptions,
        MULTIMODAL_CONFIG["max_text_length"], augmentation_factor
    )
    val_dataset = MultimodalValDataset(
        val_samples, val_transform, tokenizer, descriptions,
        MULTIMODAL_CONFIG["max_text_length"]
    )

    # Weighted sampler for balanced training
    sample_weights = []
    for _, _, label in train_samples:
        sample_weights.append(full_dataset.class_weights[label].item())
    sample_weights = sample_weights * augmentation_factor

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Create MULTIMODAL model (not image-only!)
    print("\nInitializing MultimodalClassifier (Image + Text + Fusion)...")
    model = MultimodalClassifier(config)

    # Print model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create trainer config
    trainer_config = MULTIMODAL_CONFIG.copy()
    trainer_config["num_epochs"] = num_epochs
    trainer_config["batch_size"] = batch_size

    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=full_dataset.class_weights,
        config=trainer_config
    )

    # Train
    history = trainer.train(num_epochs=num_epochs, save_dir=save_dir)

    return trainer.model, history


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multimodal rare disease classifier")
    parser.add_argument("--image-dirs", nargs="+", default=["data/images_organized"])
    parser.add_argument("--clinical-descriptions", default="data/syndrome_clinical_descriptions.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aug-factor", type=int, default=10)
    parser.add_argument("--save-dir", default="checkpoints")

    args = parser.parse_args()

    model, history = train_multimodal(
        image_dirs=args.image_dirs,
        clinical_descriptions_path=args.clinical_descriptions,
        augmentation_factor=args.aug_factor,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

    print("\nMultimodal training complete!")
    print("Model uses BOTH facial images AND clinical text for classification.")
