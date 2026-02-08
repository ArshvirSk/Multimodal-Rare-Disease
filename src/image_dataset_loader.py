"""
Image Dataset Loader for Rare Disease Facial Images.
Handles face detection, preprocessing, augmentation, and PyTorch dataset creation.
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: MTCNN not available. Face detection will be skipped.")

from .config import get_config


class FaceDetector:
    """Face detection using MTCNN (optional)."""

    def __init__(self, image_size: int = 224, margin: int = 20, device: str = "cpu"):
        """
        Initialize face detector.

        Args:
            image_size: Output image size
            margin: Margin around detected face
            device: Device to run detection on
        """
        self.image_size = image_size
        self.margin = margin
        self.device = device

        if MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=image_size,
                margin=margin,
                device=device,
                post_process=False
            )
        else:
            self.detector = None

    def detect_and_crop(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Detect face and crop image.

        Args:
            image: PIL Image

        Returns:
            Cropped face image or None if no face detected
        """
        if self.detector is None:
            # No MTCNN, return resized image
            return image.resize((self.image_size, self.image_size))

        try:
            # MTCNN returns tensor directly
            face = self.detector(image)
            if face is not None:
                # Convert tensor to PIL Image
                face = face.permute(1, 2, 0).numpy().astype(np.uint8)
                return Image.fromarray(face)
        except Exception as e:
            print(f"Face detection failed: {e}")

        return None


def get_train_transforms(config) -> transforms.Compose:
    """
    Get training data augmentation transforms.

    Args:
        config: Configuration object

    Returns:
        Composed transforms for training
    """
    transform_list = [
        transforms.Resize((config.data.image_size, config.data.image_size)),
    ]

    if config.data.augment_images:
        transform_list.extend([
            transforms.RandomHorizontalFlip(
                p=config.data.horizontal_flip_prob),
            transforms.RandomRotation(degrees=config.data.rotation_degrees),
            transforms.ColorJitter(
                brightness=config.data.brightness_factor,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(transform_list)


def get_val_transforms(config) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        config: Configuration object

    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class RareDiseaseImageDataset(Dataset):
    """
    PyTorch Dataset for rare disease facial images.

    Supports two directory structures:

    1. Organized folders (recommended):
    data/images/
    ├── Syndrome_A/
    │   ├── patient_001.jpg
    │   └── ...
    ├── Syndrome_B/
    │   └── ...

    2. Flat structure with naming convention:
    data/images/
    ├── SYN_22Q_001.png  (prefix maps to syndrome)
    ├── SYN_AS_001.png
    └── ...
    """

    # Mapping from filename prefix to syndrome name for flat structure
    PREFIX_TO_SYNDROME = {
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
        syndrome_names: List[str],
        transform: Optional[Callable] = None,
        use_face_detection: bool = False,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        use_flat_structure: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            image_dir: Root directory containing syndrome folders or flat images
            syndrome_names: List of syndrome names (folder names or target syndrome names)
            transform: Image transforms to apply
            use_face_detection: Whether to use MTCNN for face detection
            image_extensions: Valid image file extensions
            use_flat_structure: If True, expect flat dir with SYN_* prefixed files
        """
        self.image_dir = Path(image_dir)
        self.syndrome_names = syndrome_names
        self.transform = transform
        self.image_extensions = image_extensions
        self.use_flat_structure = use_flat_structure

        # Build syndrome to index mapping
        self.syndrome_to_idx = {name: idx for idx,
                                name in enumerate(syndrome_names)}
        self.idx_to_syndrome = {idx: name for name,
                                idx in self.syndrome_to_idx.items()}

        # Face detector
        self.face_detector = FaceDetector() if use_face_detection else None

        # Load all image paths and labels
        self.samples = self._load_samples()

        # Compute class weights for balanced sampling
        self.class_counts = self._compute_class_counts()
        self.class_weights = self._compute_class_weights()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their labels."""
        samples = []

        if self.use_flat_structure:
            # Load from flat directory using filename prefixes
            samples = self._load_flat_samples()
        else:
            # Load from organized folder structure
            samples = self._load_folder_samples()

        return samples

    def _load_flat_samples(self) -> List[Tuple[Path, int]]:
        """Load samples from flat directory using filename prefixes."""
        samples = []

        for img_path in self.image_dir.iterdir():
            if img_path.suffix.lower() not in self.image_extensions:
                continue

            # Try to match filename to syndrome
            filename = img_path.stem
            matched_syndrome = None

            for prefix, syndrome_name in self.PREFIX_TO_SYNDROME.items():
                if filename.startswith(prefix):
                    matched_syndrome = syndrome_name
                    break

            if matched_syndrome and matched_syndrome in self.syndrome_to_idx:
                label = self.syndrome_to_idx[matched_syndrome]
                samples.append((img_path, label))

        print(f"Loaded {len(samples)} images from flat structure")
        return samples

    def _load_folder_samples(self) -> List[Tuple[Path, int]]:
        """Load samples from organized folder structure."""
        samples = []

        for syndrome_name in self.syndrome_names:
            syndrome_dir = self.image_dir / syndrome_name

            if not syndrome_dir.exists():
                # Try with underscores instead of spaces
                syndrome_dir = self.image_dir / syndrome_name.replace(" ", "_")

            if not syndrome_dir.exists():
                print(f"Warning: Syndrome directory not found: {syndrome_dir}")
                continue

            label = self.syndrome_to_idx[syndrome_name]

            for img_path in syndrome_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    samples.append((img_path, label))

        print(
            f"Loaded {len(samples)} images from {len(self.syndrome_names)} syndromes")
        return samples

    def _compute_class_counts(self) -> Dict[int, int]:
        """Compute number of samples per class."""
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for weighted loss."""
        total = len(self.samples)
        num_classes = len(self.syndrome_names)

        weights = []
        for i in range(num_classes):
            count = self.class_counts.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for WeightedRandomSampler."""
        sample_weights = []
        for _, label in self.samples:
            weight = self.class_weights[label].item()
            sample_weights.append(weight)
        return torch.tensor(sample_weights, dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder
            image = Image.new('RGB', (224, 224), color='gray')

        # Face detection (optional)
        if self.face_detector is not None:
            detected_face = self.face_detector.detect_and_crop(image)
            if detected_face is not None:
                image = detected_face

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SyntheticImageDataset(Dataset):
    """
    Dataset for loading synthetic images generated by PDIDB StyleGAN3.

    This dataset loads pre-generated synthetic images organized by disease class.
    """

    def __init__(
        self,
        synthetic_dir: Path,
        class_mapping: Dict[str, int],
        transform: Optional[Callable] = None
    ):
        """
        Initialize synthetic image dataset.

        Args:
            synthetic_dir: Directory containing synthetic images
            class_mapping: Mapping from class name to index
            transform: Image transforms
        """
        self.synthetic_dir = Path(synthetic_dir)
        self.class_mapping = class_mapping
        self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load synthetic image samples."""
        samples = []

        for class_name, class_idx in self.class_mapping.items():
            class_dir = self.synthetic_dir / class_name

            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.png"):
                samples.append((img_path, class_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_image_dataloaders(
    image_dir: Path,
    syndrome_names: List[str],
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_weighted_sampling: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        image_dir: Root directory containing syndrome folders
        syndrome_names: List of syndrome names
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        use_weighted_sampling: Use weighted random sampling for class balance
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = get_config()

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create full dataset
    full_dataset = RareDiseaseImageDataset(
        image_dir=image_dir,
        syndrome_names=syndrome_names,
        transform=None,  # Will apply transforms separately
        use_face_detection=False
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    indices = list(range(total_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create datasets with appropriate transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_dataset = SubsetWithTransform(
        full_dataset, train_indices, train_transform)
    val_dataset = SubsetWithTransform(full_dataset, val_indices, val_transform)
    test_dataset = SubsetWithTransform(
        full_dataset, test_indices, val_transform)

    # Create samplers
    if use_weighted_sampling:
        train_sample_weights = full_dataset.get_sample_weights()[train_indices]
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_indices),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


class SubsetWithTransform(Dataset):
    """Subset of a dataset with custom transforms."""

    def __init__(self, dataset: Dataset, indices: List[int], transform: Callable):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        img_path, label = self.dataset.samples[original_idx]

        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Test the dataset loader
    config = get_config()

    print("Testing Image Dataset Loader...")
    print(f"Image size: {config.data.image_size}")
    print(f"Syndromes: {config.syndrome_names}")

    # Test transforms
    train_transform = get_train_transforms(config)
    print(f"Train transforms: {train_transform}")
