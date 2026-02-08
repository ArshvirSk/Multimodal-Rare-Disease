"""
Enhanced Data Augmentation Pipeline for Rare Disease Facial Images.
Provides aggressive augmentation to expand limited training data.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import cv2

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Using basic augmentation only.")
    print("Install with: pip install albumentations")


def get_strong_augmentation_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get aggressive augmentation transforms for limited data scenarios.

    Args:
        image_size: Target image size

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)
                          ),  # Resize larger for random crop
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # Cutout-like augmentation
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_albumentations_transforms(image_size: int = 224):
    """
    Get albumentations-based transforms for more advanced augmentation.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for this function")

    return A.Compose([
        A.Resize(image_size, image_size),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=0.1),
        ], p=0.5),
        A.OneOf([
            A.Rotate(limit=25, p=1),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=25, p=1),
            A.Affine(scale=(0.9, 1.1), translate_percent=(
                0.0, 0.1), rotate=(-20, 20), p=1),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.MotionBlur(blur_limit=5, p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.1, p=1),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.8),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
            A.Equalize(p=1),
        ], p=0.2),
        A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=16,
                            max_width=16, fill_value=0, p=1),
            A.GridDropout(ratio=0.2, p=1),
        ], p=0.2),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1),
        ], p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


class AugmentedRareDiseaseDataset(Dataset):
    """
    Dataset with aggressive augmentation and online data multiplication.
    Generates multiple augmented versions of each image per epoch.
    """

    def __init__(
        self,
        image_dir: Path,
        syndrome_names: List[str],
        augmentation_factor: int = 10,  # Generate 10x more samples
        use_albumentations: bool = True,
        image_size: int = 224,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """
        Initialize augmented dataset.

        Args:
            image_dir: Root directory containing syndrome folders
            syndrome_names: List of syndrome names (folder names)
            augmentation_factor: Number of augmented versions per original image
            use_albumentations: Use albumentations for advanced augmentation
            image_size: Target image size
            image_extensions: Valid image file extensions
        """
        self.image_dir = Path(image_dir)
        self.syndrome_names = syndrome_names
        self.augmentation_factor = augmentation_factor
        self.image_size = image_size
        self.image_extensions = image_extensions

        # Build syndrome to index mapping
        self.syndrome_to_idx = {name: idx for idx,
                                name in enumerate(syndrome_names)}

        # Setup transforms
        if use_albumentations and ALBUMENTATIONS_AVAILABLE:
            self.transform = get_albumentations_transforms(image_size)
            self.use_albumentations = True
        else:
            self.transform = get_strong_augmentation_transforms(image_size)
            self.use_albumentations = False

        # Load all image paths and labels
        self.original_samples = self._load_samples()

        # Expand samples by augmentation factor
        self.samples = self.original_samples * augmentation_factor

        print(f"Original images: {len(self.original_samples)}")
        print(f"Augmented dataset size: {len(self.samples)}")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their labels."""
        samples = []

        for syndrome_name in self.syndrome_names:
            syndrome_dir = self.image_dir / syndrome_name

            if not syndrome_dir.exists():
                print(f"Warning: Syndrome directory not found: {syndrome_dir}")
                continue

            label = self.syndrome_to_idx[syndrome_name]

            for img_path in syndrome_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    samples.append((img_path, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an augmented sample."""
        # Map to original sample
        original_idx = idx % len(self.original_samples)
        img_path, label = self.original_samples[original_idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image_np = np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.uint8)

        # Apply transforms
        if self.use_albumentations:
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']
        else:
            image = Image.fromarray(image_np)
            image_tensor = self.transform(image)

        return image_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced loss."""
        class_counts = {}
        for _, label in self.original_samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        total = len(self.original_samples)
        num_classes = len(self.syndrome_names)

        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


class MixupDataset(Dataset):
    """
    Dataset wrapper that applies mixup augmentation.
    Combines two images and their labels with a random ratio.
    """

    def __init__(self, base_dataset: Dataset, alpha: float = 0.4):
        """
        Initialize mixup dataset.

        Args:
            base_dataset: Base dataset to wrap
            alpha: Mixup alpha parameter (higher = more mixing)
        """
        self.base_dataset = base_dataset
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """Get a mixup sample."""
        # Get first sample
        img1, label1 = self.base_dataset[idx]

        # Get random second sample
        idx2 = random.randint(0, len(self.base_dataset) - 1)
        img2, label2 = self.base_dataset[idx2]

        # Mixup ratio
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix images
        mixed_img = lam * img1 + (1 - lam) * img2

        # Return mixed sample with both labels and lambda
        return mixed_img, label1, label2, lam


def create_augmented_dataloaders(
    image_dir: Path,
    syndrome_names: List[str],
    batch_size: int = 16,
    augmentation_factor: int = 10,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with heavy augmentation.

    Args:
        image_dir: Root directory containing syndrome folders
        syndrome_names: List of syndrome names
        batch_size: Batch size
        augmentation_factor: Number of augmented versions per image
        num_workers: Number of data loading workers
        train_ratio: Training set ratio
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create full augmented dataset
    full_dataset = AugmentedRareDiseaseDataset(
        image_dir=image_dir,
        syndrome_names=syndrome_names,
        augmentation_factor=augmentation_factor
    )

    # Split dataset
    total_original = len(full_dataset.original_samples)
    train_size = int(total_original * train_ratio)

    # Shuffle original indices
    indices = list(range(total_original))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Expand indices for augmented dataset
    train_aug_indices = []
    for i in train_indices:
        for j in range(augmentation_factor):
            train_aug_indices.append(i + j * total_original)

    # Validation uses no augmentation - just original indices
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    train_dataset = torch.utils.data.Subset(full_dataset, train_aug_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the augmentation pipeline
    from config import get_config

    config = get_config()

    print("Testing Enhanced Augmentation Pipeline...")
    print(f"Image size: {config.data.image_size}")

    # Test augmented dataset
    dataset = AugmentedRareDiseaseDataset(
        image_dir=Path("data/images_organized"),
        syndrome_names=[s.replace(" ", "_") for s in config.syndrome_names],
        augmentation_factor=10
    )

    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Label: {label}")
        print(f"Class weights: {dataset.get_class_weights()}")
