"""
Script to reorganize flat image files into syndrome folders.
Also provides utilities for generating more synthetic images using PDIDB.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

# Mapping from image prefix to full syndrome name
SYNDROME_MAPPING = {
    "SYN_22Q": "22q11.2_Deletion_Syndrome",
    "SYN_AS": "Angelman_Syndrome",
    "SYN_CdLS": "Cornelia_de_Lange_Syndrome",
    "SYN_KBG": "KBG_Syndrome",
    "SYN_KS": "Kabuki_Syndrome",
    "SYN_NBS": "Nicolaides_Baraitser_Syndrome",
    "SYN_NS": "Noonan_Syndrome",
    "SYN_RSTS": "Rubinstein_Taybi_Syndrome",
    "SYN_SMS": "Smith_Magenis_Syndrome",
    "SYN_WBS": "Williams_Beuren_Syndrome",
}

# Mapping to class index (matching config.py order)
SYNDROME_TO_IDX = {
    "Cornelia_de_Lange_Syndrome": 0,
    "Williams_Beuren_Syndrome": 1,
    "Noonan_Syndrome": 2,
    "Kabuki_Syndrome": 3,
    "KBG_Syndrome": 4,
    "Angelman_Syndrome": 5,
    "Rubinstein_Taybi_Syndrome": 6,
    "Smith_Magenis_Syndrome": 7,
    "Nicolaides_Baraitser_Syndrome": 8,
    "22q11.2_Deletion_Syndrome": 9,
}


def reorganize_images(
    source_dir: str = "data/images",
    dest_dir: str = "data/images_organized",
    copy: bool = True
):
    """
    Reorganize flat image files into syndrome-specific folders.

    Args:
        source_dir: Directory containing flat image files
        dest_dir: Destination directory for organized structure
        copy: If True, copy files; if False, move files
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)

    # Create syndrome folders
    for syndrome_name in SYNDROME_MAPPING.values():
        (dest_path / syndrome_name).mkdir(exist_ok=True)

    # Process each image file
    moved_count = 0
    for img_file in source_path.glob("*.png"):
        # Extract syndrome prefix from filename
        filename = img_file.stem  # e.g., "SYN_22Q_001"

        # Find matching syndrome
        syndrome_folder = None
        for prefix, folder_name in SYNDROME_MAPPING.items():
            if filename.startswith(prefix):
                syndrome_folder = folder_name
                break

        if syndrome_folder:
            dest_file = dest_path / syndrome_folder / img_file.name
            if copy:
                shutil.copy2(img_file, dest_file)
            else:
                shutil.move(img_file, dest_file)
            moved_count += 1
            print(
                f"{'Copied' if copy else 'Moved'}: {img_file.name} -> {syndrome_folder}/")

    print(f"\nTotal images organized: {moved_count}")
    print(f"Destination: {dest_path.absolute()}")

    # Print summary
    print("\nImages per syndrome:")
    for syndrome_name in SYNDROME_MAPPING.values():
        count = len(list((dest_path / syndrome_name).glob("*.png")))
        print(f"  {syndrome_name}: {count}")

    return dest_path


def get_syndrome_class_mapping() -> Dict[str, int]:
    """Get mapping from folder names to class indices."""
    return SYNDROME_TO_IDX


def generate_synthetic_images_command(
    network_pkl: str,
    class_idx: int,
    num_samples: int,
    output_dir: str,
    truncation: float = 0.7
) -> str:
    """
    Generate the command to create synthetic images using PDIDB.

    Args:
        network_pkl: Path to the trained StyleGAN3 network pickle
        class_idx: Class index for conditional generation
        num_samples: Number of images to generate
        output_dir: Output directory
        truncation: Truncation psi (0.5-1.0, lower = more consistent)

    Returns:
        Command string to run
    """
    cmd = (
        f"python PDIDB/gen_images.py "
        f"--network={network_pkl} "
        f"--samples={num_samples} "
        f"--class={class_idx} "
        f"--trunc={truncation} "
        f"--outdir={output_dir}"
    )
    return cmd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reorganize images into syndrome folders")
    parser.add_argument("--source", default="data/images",
                        help="Source directory")
    parser.add_argument(
        "--dest", default="data/images_organized", help="Destination directory")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying")

    args = parser.parse_args()

    reorganize_images(
        source_dir=args.source,
        dest_dir=args.dest,
        copy=not args.move
    )
