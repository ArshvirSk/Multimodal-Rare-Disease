"""
Verify the optimized training setup for small datasets.
Run this to check if all components are working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("="*60)
    print("VERIFICATION: Small Dataset Training Setup")
    print("="*60)

    # 1. Check imports
    print("\n1. Checking imports...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"   ✗ PyTorch import failed: {e}")
        return

    try:
        from torchvision import transforms
        print("   ✓ torchvision")
    except ImportError as e:
        print(f"   ✗ torchvision import failed: {e}")
        return

    # 2. Check configuration
    print("\n2. Checking configuration...")
    try:
        from src.config import get_config, Config
        config = get_config()
        print(f"   ✓ Config loaded")
        print(f"   ✓ Batch size: {config.training.batch_size}")
        print(f"   ✓ Learning rate: {config.training.learning_rate}")
        print(f"   ✓ Freeze layers: {config.cnn_encoder.freeze_layers}")
        print(f"   ✓ Dropout: {config.cnn_encoder.dropout}")
        print(f"   ✓ Syndrome classes: {len(config.syndrome_names)}")
    except Exception as e:
        print(f"   ✗ Config loading failed: {e}")
        return

    # 3. Check image data
    print("\n3. Checking image data...")
    organized_dir = Path("data/images_organized")
    if organized_dir.exists():
        syndrome_folders = list(organized_dir.iterdir())
        total_images = 0
        for folder in syndrome_folders:
            if folder.is_dir():
                images = list(folder.glob("*.png")) + \
                    list(folder.glob("*.jpg"))
                total_images += len(images)
                print(f"   ✓ {folder.name}: {len(images)} images")
        print(f"   ✓ Total images: {total_images}")
    else:
        print("   ✗ data/images_organized not found")
        print("   → Run scripts/reorganize_images.py first")
        return

    # 4. Check model creation
    print("\n4. Checking model creation...")
    try:
        from src.multimodal_classifier import ImageOnlyClassifier
        model = ImageOnlyClassifier(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        print(f"   ✓ ImageOnlyClassifier created")
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
        print(
            f"   ✓ Frozen parameters: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Check data augmentation
    print("\n5. Checking data augmentation transforms...")
    try:
        from src.train_small_data import get_heavy_augmentation_transforms, get_val_transforms
        train_transform = get_heavy_augmentation_transforms(224)
        val_transform = get_val_transforms(224)
        print(
            f"   ✓ Training transforms: {len(train_transform.transforms)} operations")
        print(
            f"   ✓ Validation transforms: {len(val_transform.transforms)} operations")
    except Exception as e:
        print(f"   ✗ Transform creation failed: {e}")
        return

    # 6. Test dataset loading
    print("\n6. Testing dataset loading...")
    try:
        from src.train_small_data import SmallDatasetAugmented
        dataset = SmallDatasetAugmented(
            image_dirs=[organized_dir],
            syndrome_names=config.syndrome_names,
            transform=train_transform,
            augmentation_factor=20,
            is_training=True
        )
        print(f"   ✓ Dataset created")
        print(f"   ✓ Original samples: {len(dataset.original_samples)}")
        print(f"   ✓ Augmented samples: {len(dataset)}")

        # Test loading a sample
        if len(dataset) > 0:
            sample_img, sample_label = dataset[0]
            print(f"   ✓ Sample shape: {sample_img.shape}")
            print(f"   ✓ Sample label: {sample_label}")
    except Exception as e:
        print(f"   ✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Quick forward pass test
    print("\n7. Testing forward pass...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Create dummy batch
        dummy_batch = torch.randn(2, 3, 224, 224).to(device)

        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)

        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Output logits shape: {output['logits'].shape}")
        print(f"   ✓ Output probs shape: {output['probs'].shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE - All checks passed! ✓")
    print("="*60)
    print("\nReady to train! Run:")
    print("  python run_training.py --epochs 50 --aug-factor 20")
    print("\nOptimizations applied:")
    print(
        f"  • Data augmentation: {total_images} → {total_images * 20} samples")
    print(
        f"  • Transfer learning: {100*(total_params-trainable_params)/total_params:.1f}% backbone frozen")
    print(f"  • Label smoothing: {config.training.label_smoothing}")
    print(f"  • Dropout regularization: {config.cnn_encoder.dropout}")


if __name__ == "__main__":
    main()
