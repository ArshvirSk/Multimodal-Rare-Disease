"""
Training Pipeline for Multimodal Rare Disease Classifier.
Supports multimodal and unimodal training modes with class imbalance handling.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import get_config, Config
from .multimodal_classifier import (
    MultimodalClassifier,
    ImageOnlyClassifier,
    TextOnlyClassifier,
)


def get_safe_device(requested_device: str = "cuda") -> str:
    """
    Safely determine the device to use.

    Args:
        requested_device: Requested device (cuda or cpu)

    Returns:
        Safe device string that will work
    """
    if requested_device == "cuda":
        # Check if CUDA is truly available (compiled with CUDA + driver present)
        try:
            if torch.cuda.is_available():
                # Try to actually use CUDA to verify it works
                torch.zeros(1).cuda()
                return "cuda"
        except Exception:
            pass
        print("CUDA not available, falling back to CPU")
        return "cpu"
    return requested_device


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Trainer for multimodal rare disease classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Config] = None,
        mode: str = "multimodal",
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            mode: Training mode (multimodal, image_only, text_only)
        """
        if config is None:
            config = get_config()

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mode = mode

        # Device - use safe device detection
        safe_device = get_safe_device(config.training.device)
        self.device = torch.device(safe_device)
        self.model.to(self.device)

        # Loss function with class weights
        self.criterion = self._setup_criterion()

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler
        self.scheduler = self._setup_scheduler()

        # Early stopping
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.patience,
                min_delta=config.training.min_delta,
                mode="min",
            )
        else:
            self.early_stopping = None

        # Mixed precision
        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _setup_criterion(self) -> nn.Module:
        """Setup loss function with class weights."""
        # Get class weights from data loader if available
        if hasattr(self.train_loader.dataset, "class_weights"):
            weights = self.train_loader.dataset.class_weights.to(self.device)
            return nn.CrossEntropyLoss(weight=weights)

        return nn.CrossEntropyLoss()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        params = self.model.parameters()

        if self.config.training.optimizer == "adam":
            return Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adamw":
            return AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "sgd":
            return SGD(
                params,
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            return AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.training.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs, eta_min=1e-7
            )
        elif self.config.training.scheduler == "step":
            return StepLR(
                self.optimizer, step_size=30, gamma=self.config.training.lr_decay_factor
            )
        elif self.config.training.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.training.lr_decay_factor,
                patience=5,
            )
        else:
            return None

    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            self.optimizer.zero_grad()

            # Move data to device
            if self.mode == "multimodal":
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        output = self.model(images, input_ids, attention_mask)
                        loss = self.criterion(output["logits"], labels)
                else:
                    output = self.model(images, input_ids, attention_mask)
                    loss = self.criterion(output["logits"], labels)

                predictions = output["logits"].argmax(dim=-1)

            elif self.mode == "image_only":
                images = (
                    batch[0].to(self.device)
                    if isinstance(batch, tuple)
                    else batch["image"].to(self.device)
                )
                labels = (
                    batch[1].to(self.device)
                    if isinstance(batch, tuple)
                    else batch["label"].to(self.device)
                )

                if self.use_amp:
                    with autocast():
                        output = self.model(images)
                        loss = self.criterion(output["logits"], labels)
                else:
                    output = self.model(images)
                    loss = self.criterion(output["logits"], labels)

                predictions = output["logits"].argmax(dim=-1)

            elif self.mode == "text_only":
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                if self.use_amp:
                    with autocast():
                        output = self.model(input_ids, attention_mask)
                        loss = self.criterion(output["logits"], labels)
                else:
                    output = self.model(input_ids, attention_mask)
                    loss = self.criterion(output["logits"], labels)

                predictions = output["logits"].argmax(dim=-1)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_val
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.gradient_clip_val > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_val
                    )
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move data to device
            if self.mode == "multimodal":
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                output = self.model(images, input_ids, attention_mask)
                loss = self.criterion(output["logits"], labels)
                predictions = output["logits"].argmax(dim=-1)

            elif self.mode == "image_only":
                images = (
                    batch[0].to(self.device)
                    if isinstance(batch, tuple)
                    else batch["image"].to(self.device)
                )
                labels = (
                    batch[1].to(self.device)
                    if isinstance(batch, tuple)
                    else batch["label"].to(self.device)
                )

                output = self.model(images)
                loss = self.criterion(output["logits"], labels)
                predictions = output["logits"].argmax(dim=-1)

            elif self.mode == "text_only":
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                output = self.model(input_ids, attention_mask)
                loss = self.criterion(output["logits"], labels)
                predictions = output["logits"].argmax(dim=-1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": {
                "mode": self.mode,
                "num_classes": self.config.classifier.num_classes,
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save last checkpoint
        checkpoint_dir = self.config.training.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{self.mode}_last.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / f"{self.mode}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)

        return checkpoint["epoch"]

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Run training loop.

        Args:
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        print(f"\nStarting {self.mode} training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print("-" * 50)

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_acc = self._validate()

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

            # Save checkpoint
            if self.config.training.save_best_only:
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch, is_best=is_best)

            epoch_time = time.time() - epoch_start

            # Print progress
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
                + (" *" if is_best else "")
            )

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(
            f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}"
        )

        return self.history


def train_model(
    mode: str = "multimodal",
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    config: Optional[Config] = None,
    num_epochs: int = 100,
) -> Tuple[nn.Module, Dict]:
    """
    Train a model with the specified mode.

    Args:
        mode: Training mode (multimodal, image_only, text_only)
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        num_epochs: Number of training epochs

    Returns:
        Tuple of (trained_model, history)
    """
    if config is None:
        config = get_config()

    # Create model based on mode
    if mode == "multimodal":
        model = MultimodalClassifier(config)
    elif mode == "image_only":
        model = ImageOnlyClassifier(config)
    elif mode == "text_only":
        model = TextOnlyClassifier(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        mode=mode,
    )

    # Train
    history = trainer.train(num_epochs)

    return model, history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train multimodal rare disease classifier"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "image_only", "text_only"],
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--smoke_test", action="store_true", help="Run a quick smoke test"
    )

    args = parser.parse_args()

    # Setup configuration
    config = get_config()
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    # Use safe device detection to handle CPU-only PyTorch
    config.training.device = get_safe_device(args.device)

    if args.smoke_test:
        # Quick smoke test with dummy data
        print("Running smoke test with dummy data (2 epochs)...")
        config.training.num_epochs = 2

        dummy_images = torch.randn(32, 3, 224, 224)
        dummy_input_ids = torch.randint(0, 28000, (32, 128))
        dummy_attention = torch.ones(32, 128, dtype=torch.long)
        dummy_labels = torch.randint(0, 10, (32,))

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 32

            def __getitem__(self, idx):
                return {
                    "image": dummy_images[idx],
                    "input_ids": dummy_input_ids[idx],
                    "attention_mask": dummy_attention[idx],
                    "label": dummy_labels[idx],
                }

        dataset = DummyDataset()
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=8)
    else:
        # Load real FGDD data
        print(
            f"Loading real data from FGDD dataset for {config.training.num_epochs} epochs..."
        )

        import pandas as pd
        from pathlib import Path
        from transformers import AutoTokenizer
        from sklearn.model_selection import train_test_split

        # Paths
        fgdd_dir = config.data.fgdd_dir
        fgdd_csv = fgdd_dir / "FGDD" / "FGDD.csv"
        phenotype_csv = fgdd_dir / "Raw data" / "phenotype.csv"

        if not fgdd_csv.exists():
            print(f"ERROR: FGDD data not found at {fgdd_csv}")
            raise FileNotFoundError(f"Please download FGDD dataset to {fgdd_dir}")

        print(f"Found FGDD data at: {fgdd_csv}")

        # Load main FGDD data
        fgdd_df = pd.read_csv(fgdd_csv)
        print(
            f"Loaded {len(fgdd_df)} patient samples with {len(fgdd_df.columns)} features"
        )

        # Load phenotype names for HPO IDs
        phenotype_names = {}
        if phenotype_csv.exists():
            phenotype_df = pd.read_csv(phenotype_csv)
            phenotype_names = dict(
                zip(phenotype_df["Pid"], phenotype_df["phenotype_name"])
            )

        # Get disease distribution - use top 10 diseases
        disease_counts = fgdd_df["Disease_name"].value_counts().head(10)
        disease_to_label = {name: i for i, name in enumerate(disease_counts.index)}

        print(f"Using top 10 diseases: {list(disease_to_label.keys())}")

        # Filter to top 10 diseases
        fgdd_df = fgdd_df[fgdd_df["Disease_name"].isin(disease_to_label.keys())]

        # Create text samples from HPO phenotypes
        hpo_columns = [col for col in fgdd_df.columns if col.startswith("HP:")]
        print(f"Found {len(hpo_columns)} HPO phenotype columns")

        samples = []
        for _, row in fgdd_df.iterrows():
            # Get active phenotypes (value = 1)
            active_hpos = [col for col in hpo_columns if row.get(col, 0) == 1]

            # Build description from phenotype names
            phenotypes = []
            for hpo_id in active_hpos[:5]:  # Limit to 5 phenotypes per sample
                if hpo_id in phenotype_names:
                    phenotypes.append(phenotype_names[hpo_id])
                else:
                    phenotypes.append(hpo_id)

            if phenotypes:
                text = f"Patient presents with: {', '.join(phenotypes)}. Suspected: {row['Disease_name']}."
                samples.append(
                    {"text": text, "label": disease_to_label[row["Disease_name"]]}
                )

        print(f"Created {len(samples)} text samples from FGDD patient data")

        if len(samples) < 10:
            raise ValueError("Not enough samples in FGDD dataset")

        # Load PDIDB images
        pdidb_image_dir = config.data.fgdd_dir.parent / "images"
        pdidb_metadata_path = config.data.fgdd_dir.parent / "phenotype_metadata.csv"

        pdidb_images = {}  # disease_name -> list of image tensors
        pdidb_disease_to_label = {}

        if pdidb_metadata_path.exists():
            print(f"\nLoading PDIDB images from: {pdidb_image_dir}")
            pdidb_df = pd.read_csv(pdidb_metadata_path)

            # Image transforms with augmentation
            from torchvision import transforms
            from PIL import Image

            augment_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            base_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Create disease label mapping for PDIDB
            pdidb_diseases = pdidb_df["Disease_Type"].unique()
            pdidb_disease_to_label = {d: i for i, d in enumerate(pdidb_diseases)}
            print(f"PDIDB diseases ({len(pdidb_diseases)}): {list(pdidb_diseases)}")

            # Load and augment images
            for _, row in pdidb_df.iterrows():
                img_path = pdidb_image_dir / f"{row['Image_ID']}.png"
                if img_path.exists():
                    disease = row["Disease_Type"]
                    if disease not in pdidb_images:
                        pdidb_images[disease] = []

                    img = Image.open(img_path).convert("RGB")

                    # Add original + 4 augmented versions = 5x data
                    pdidb_images[disease].append(base_transform(img))
                    for _ in range(4):
                        pdidb_images[disease].append(augment_transform(img))

            total_img_samples = sum(len(v) for v in pdidb_images.values())
            print(
                f"Loaded {total_img_samples} image samples (with augmentation) across {len(pdidb_images)} diseases"
            )

        # Create image-only dataset for multimodal training
        image_samples = []
        if pdidb_images:
            for disease, img_list in pdidb_images.items():
                label = pdidb_disease_to_label[disease]
                for img_tensor in img_list:
                    image_samples.append(
                        {"image": img_tensor, "label": label, "disease": disease}
                    )
            print(f"Created {len(image_samples)} image training samples")

        # Tokenize FGDD texts
        tokenizer = AutoTokenizer.from_pretrained(config.text_encoder.model_name)

        texts = [s["text"] for s in samples]
        text_labels = [s["label"] for s in samples]

        encodings = tokenizer(
            texts,
            max_length=config.data.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_masks = encodings["attention_mask"]
        text_labels_tensor = torch.tensor(text_labels, dtype=torch.long)

        # For multimodal: pair images with random text samples (since diseases don't overlap)
        # For image_only/text_only: use respective data
        if image_samples:
            # Stack all images and create combined dataset
            all_images = torch.stack([s["image"] for s in image_samples])
            all_image_labels = torch.tensor(
                [s["label"] for s in image_samples], dtype=torch.long
            )

            # Use image dataset size, pair with random/cycled text
            n_img = len(image_samples)
            text_indices = list(range(len(samples))) * (n_img // len(samples) + 1)
            text_indices = text_indices[:n_img]

            combined_input_ids = input_ids[text_indices]
            combined_attention = attention_masks[text_indices]
            combined_labels = all_image_labels  # Use image labels for now

            # Train/val split
            train_idx, val_idx = train_test_split(
                range(n_img),
                test_size=0.2,
                random_state=42,
                stratify=combined_labels.tolist(),
            )

            class MultimodalDataset(torch.utils.data.Dataset):
                def __init__(self, indices):
                    self.indices = indices

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, idx):
                    i = self.indices[idx]
                    return {
                        "image": all_images[i],
                        "input_ids": combined_input_ids[i],
                        "attention_mask": combined_attention[i],
                        "label": combined_labels[i],
                    }

            train_dataset = MultimodalDataset(train_idx)
            val_dataset = MultimodalDataset(val_idx)
        else:
            # No images - fallback to text-only with placeholder images
            images = torch.randn(len(samples), 3, 224, 224)
            train_idx, val_idx = train_test_split(
                range(len(samples)),
                test_size=0.2,
                random_state=42,
                stratify=text_labels,
            )

            class FGDDDataset(torch.utils.data.Dataset):
                def __init__(self, indices):
                    self.indices = indices

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, idx):
                    i = self.indices[idx]
                    return {
                        "image": images[i],
                        "input_ids": input_ids[i],
                        "attention_mask": attention_masks[i],
                        "label": text_labels_tensor[i],
                    }

            train_dataset = FGDDDataset(train_idx)
            val_dataset = FGDDDataset(val_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Train model
    model, history = train_model(
        mode=args.mode,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_epochs=config.training.num_epochs,
    )

    print(f"\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
