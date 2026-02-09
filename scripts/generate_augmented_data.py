"""
Generate Augmented Images to Increase Dataset Size.
Creates multiple augmented versions of each original image and saves them to disk.
This effectively increases your dataset from 50 to 500+ images.
"""

import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm


def augment_image(image: Image.Image, aug_type: int) -> Image.Image:
    """
    Apply a specific augmentation to an image.

    Args:
        image: PIL Image
        aug_type: Augmentation type (0-9)

    Returns:
        Augmented PIL Image
    """
    width, height = image.size

    if aug_type == 0:
        # Horizontal flip
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    elif aug_type == 1:
        # Slight rotation left
        return image.rotate(random.uniform(-15, -5), resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    elif aug_type == 2:
        # Slight rotation right
        return image.rotate(random.uniform(5, 15), resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    elif aug_type == 3:
        # Brightness increase
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(random.uniform(1.1, 1.3))

    elif aug_type == 4:
        # Brightness decrease
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(random.uniform(0.7, 0.9))

    elif aug_type == 5:
        # Contrast increase
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(random.uniform(1.1, 1.4))

    elif aug_type == 6:
        # Saturation change
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(random.uniform(0.8, 1.2))

    elif aug_type == 7:
        # Slight zoom (center crop + resize)
        crop_percent = random.uniform(0.85, 0.95)
        new_w = int(width * crop_percent)
        new_h = int(height * crop_percent)
        left = (width - new_w) // 2
        top = (height - new_h) // 2
        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((width, height), Image.BILINEAR)

    elif aug_type == 8:
        # Slight blur
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    elif aug_type == 9:
        # Flip + rotation
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        return flipped.rotate(random.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    return image


def generate_augmented_dataset(
    source_dir: Path,
    output_dir: Path,
    num_augmentations: int = 10
):
    """
    Generate augmented images from source directory.
    
    Supports two formats:
    1. Flat structure: data/images/SYN_CODE_001.png (extracts syndrome from filename)
    2. Folder structure: data/images/SYN_CODE/001.png (uses folder name)

    Args:
        source_dir: Directory with images (flat or with syndrome subfolders)
        output_dir: Output directory for augmented images
        num_augmentations: Number of augmented versions per original image
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per image: {num_augmentations}")
    print()

    total_original = 0
    total_generated = 0
    
    # Check if source has subfolders or flat files
    has_subfolders = any(p.is_dir() for p in source_dir.iterdir())
    
    if has_subfolders:
        # Process each syndrome folder
        for syndrome_folder in source_dir.iterdir():
            if not syndrome_folder.is_dir():
                continue

            # Create output folder for this syndrome
            output_syndrome_dir = output_dir / syndrome_folder.name
            output_syndrome_dir.mkdir(parents=True, exist_ok=True)

            # Get all images in this folder
            images = list(syndrome_folder.glob("*.png")) + \
                list(syndrome_folder.glob("*.jpg"))

            print(
                f"Processing {syndrome_folder.name}: {len(images)} original images")

            for img_path in tqdm(images, desc=f"  {syndrome_folder.name}"):
                orig, gen = process_single_image(
                    img_path, output_syndrome_dir, num_augmentations)
                total_original += orig
                total_generated += gen
    else:
        # Flat structure - extract syndrome from filename (e.g., SYN_CdLS_001.png)
        images = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
        
        # Group by syndrome code
        syndrome_images = {}
        for img_path in images:
            # Extract syndrome code from filename like SYN_CdLS_001.png
            parts = img_path.stem.split("_")
            if len(parts) >= 2:
                syndrome_code = f"{parts[0]}_{parts[1]}"  # e.g., "SYN_CdLS"
            else:
                syndrome_code = "unknown"
            
            if syndrome_code not in syndrome_images:
                syndrome_images[syndrome_code] = []
            syndrome_images[syndrome_code].append(img_path)
        
        print(f"Found {len(images)} images across {len(syndrome_images)} syndromes")
        print()
        
        for syndrome_code, img_list in syndrome_images.items():
            output_syndrome_dir = output_dir / syndrome_code
            output_syndrome_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing {syndrome_code}: {len(img_list)} original images")
            
            for img_path in tqdm(img_list, desc=f"  {syndrome_code}"):
                orig, gen = process_single_image(
                    img_path, output_syndrome_dir, num_augmentations)
                total_original += orig
                total_generated += gen

    print()
    print("=" * 60)
    print(f"DATASET EXPANSION COMPLETE")
    print("=" * 60)
    print(f"Original images: {total_original}")
    print(f"Total images generated: {total_generated}")
    print(f"Expansion factor: {total_generated / max(total_original, 1):.1f}x")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


def process_single_image(img_path: Path, output_dir: Path, num_augmentations: int):
    """Process a single image and generate augmentations."""
    total_original = 0
    total_generated = 0
    
    try:
        # Load original image
        original = Image.open(img_path).convert('RGB')
        total_original += 1

        # Save original (copy)
        original_name = img_path.stem
        original.save(output_dir / f"{original_name}_orig.png")
        total_generated += 1

        # Generate augmented versions
        for aug_idx in range(num_augmentations):
            aug_type = aug_idx % 10  # Cycle through augmentation types

            # Apply augmentation
            augmented = augment_image(original, aug_type)

            # Save augmented image
            aug_name = f"{original_name}_aug{aug_idx:02d}.png"
            augmented.save(output_dir / aug_name)
            total_generated += 1

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    
    return total_original, total_generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate augmented images to increase dataset size")
    parser.add_argument(
        "--source", default="data/images", help="Source image directory")
    parser.add_argument(
        "--output", default="data/images_augmented", help="Output directory")
    parser.add_argument("--num-aug", type=int, default=10,
                        help="Number of augmentations per image")

    args = parser.parse_args()

    generate_augmented_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        num_augmentations=args.num_aug
    )
