"""
Run optimized training for small datasets.
This script provides the best accuracy for limited image data scenarios.
"""

from src.train_small_data import train_with_small_data
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train rare disease classifier with small data optimization"
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        default=["data/images_augmented"],
        help="Directories containing images"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (small for limited data)"
    )
    parser.add_argument(
        "--aug-factor",
        type=int,
        default=20,
        help="Augmentation factor (multiply dataset size)"
    )
    parser.add_argument(
        "--save-dir",
        default="checkpoints",
        help="Directory to save model checkpoints"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("RARE DISEASE CLASSIFIER - SMALL DATA TRAINING")
    print("="*60)
    print("\nKey optimizations for limited data:")
    print("  ✓ Heavy data augmentation (20x by default)")
    print("  ✓ Transfer learning with frozen backbone layers")
    print("  ✓ Label smoothing for better generalization")
    print("  ✓ Mixup augmentation for robustness")
    print("  ✓ Weighted sampling for class balance")
    print("  ✓ Cosine annealing with warm restarts")
    print("  ✓ Aggressive dropout regularization")
    print("="*60 + "\n")

    model, history = train_with_small_data(
        image_dirs=args.image_dirs,
        augmentation_factor=args.aug_factor,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

    print("\n✓ Training complete! Best model saved to:", args.save_dir)
