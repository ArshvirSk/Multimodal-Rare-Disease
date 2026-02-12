"""
Run Multimodal Training Script.
Easy-to-use script to train the multimodal rare disease classifier.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train_multimodal import train_multimodal


def main():
    """Run multimodal training with sensible defaults."""
    print("=" * 70)
    print("MULTIMODAL RARE DISEASE CLASSIFIER")
    print("Training with Facial Images + Clinical Text Fusion")
    print("=" * 70)
    
    # Check for required files
    image_dirs = ["data/images_augmented", "data/images_organized"]
    clinical_path = Path("data/syndrome_clinical_descriptions.json")
    
    image_dir_found = None
    for d in image_dirs:
        if Path(d).exists():
            image_dir_found = d
            break
    
    if not image_dir_found:
        print(f"\n❌ ERROR: No image directory found!")
        print(f"   Looked for: {image_dirs}")
        print(f"   Please ensure your images are organized in one of these directories.")
        return
    
    if not clinical_path.exists():
        print(f"\n❌ ERROR: Clinical descriptions not found at {clinical_path}")
        return
    
    print(f"\n✓ Image directory: {image_dir_found}")
    print(f"✓ Clinical descriptions: {clinical_path}")
    print()
    
    # Run training
    model, history = train_multimodal(
        image_dirs=image_dirs,
        clinical_descriptions_path=str(clinical_path),
        augmentation_factor=10,  # 10x augmentation
        num_epochs=60,
        batch_size=8,
        save_dir="checkpoints"
    )
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nSaved models:")
    print("  - checkpoints/multimodal_best.pt  (best validation accuracy)")
    print("  - checkpoints/multimodal_last.pt  (final epoch)")
    print("\nThis model uses TRUE MULTIMODAL FUSION:")
    print("  • CNNEncoder (ResNet50) for facial image features")
    print("  • TextEncoder (BioBERT) for clinical text features")
    print("  • Attention-based fusion combining both modalities")
    print("=" * 70)


if __name__ == "__main__":
    main()
