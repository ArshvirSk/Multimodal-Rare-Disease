"""
Synthetic Image Generation Pipeline using PDIDB StyleGAN3.
Generates diverse synthetic facial images for rare disease classes.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import argparse

import numpy as np
import torch
from tqdm import tqdm

# Add PDIDB to path
sys.path.insert(0, str(Path(__file__).parent.parent / "PDIDB"))

try:
    import dnnlib
    import legacy
    PDIDB_AVAILABLE = True
except ImportError:
    PDIDB_AVAILABLE = False
    print("Warning: PDIDB modules not available. Run from project root.")


# Class mapping from syndrome to StyleGAN class index
# This should match the training configuration of your StyleGAN model
SYNDROME_CLASS_MAPPING = {
    "22q11.2_Deletion_Syndrome": 0,
    "Angelman_Syndrome": 1,
    "Cornelia_de_Lange_Syndrome": 2,
    "KBG_Syndrome": 3,
    "Kabuki_Syndrome": 4,
    "Nicolaides_Baraitser_Syndrome": 5,
    "Noonan_Syndrome": 6,
    "Rubinstein_Taybi_Syndrome": 7,
    "Smith_Magenis_Syndrome": 8,
    "Williams_Beuren_Syndrome": 9,
}


class SyntheticImageGenerator:
    """
    Generator for synthetic rare disease facial images using StyleGAN3.
    """

    def __init__(
        self,
        network_pkl: str,
        device: str = "cuda",
        truncation_psi: float = 0.7
    ):
        """
        Initialize the generator.

        Args:
            network_pkl: Path to trained StyleGAN3 network pickle
            device: Device to run generation on
            truncation_psi: Truncation parameter (0.5-1.0, lower = more consistent)
        """
        if not PDIDB_AVAILABLE:
            raise ImportError("PDIDB modules not available")

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.truncation_psi = truncation_psi

        print(f"Loading network from {network_pkl}...")
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        self.z_dim = self.G.z_dim
        self.c_dim = self.G.c_dim  # Number of classes

        print(f"Generator loaded. z_dim={self.z_dim}, c_dim={self.c_dim}")

    def generate_images(
        self,
        class_idx: int,
        num_images: int,
        output_dir: str,
        prefix: str = "synthetic",
        seed: Optional[int] = None,
        noise_mode: str = 'const'
    ) -> List[Path]:
        """
        Generate synthetic images for a specific class.

        Args:
            class_idx: Class index for conditional generation
            num_images: Number of images to generate
            output_dir: Output directory
            prefix: Filename prefix
            seed: Random seed (None for random)
            noise_mode: Noise mode ('const', 'random', 'none')

        Returns:
            List of generated image paths
        """
        import PIL.Image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare class label
        label = torch.zeros([1, self.c_dim], device=self.device)
        if self.c_dim > 0:
            label[:, class_idx] = 1

        generated_paths = []

        for i in tqdm(range(num_images), desc=f"Generating class {class_idx}"):
            # Generate random latent vector
            z = torch.randn([1, self.z_dim], device=self.device)

            # Generate image
            with torch.no_grad():
                img = self.G(
                    z, label, truncation_psi=self.truncation_psi, noise_mode=noise_mode)

            # Convert to PIL Image
            img = (img.permute(0, 2, 3, 1) * 127.5 +
                   128).clamp(0, 255).to(torch.uint8)
            img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

            # Save image
            filename = f"{prefix}_{class_idx:02d}_{i:04d}.png"
            filepath = output_path / filename
            img_pil.save(filepath)
            generated_paths.append(filepath)

        print(f"Generated {num_images} images in {output_path}")
        return generated_paths

    def generate_dataset(
        self,
        output_dir: str,
        images_per_class: int = 100,
        class_indices: Optional[List[int]] = None,
        seed: int = 42
    ) -> Dict[int, List[Path]]:
        """
        Generate a full synthetic dataset for all classes.

        Args:
            output_dir: Output directory
            images_per_class: Number of images per class
            class_indices: Specific class indices to generate (None for all)
            seed: Random seed

        Returns:
            Dictionary mapping class index to list of image paths
        """
        if class_indices is None:
            class_indices = list(range(self.c_dim))

        torch.manual_seed(seed)
        np.random.seed(seed)

        output_path = Path(output_dir)
        all_generated = {}

        for class_idx in class_indices:
            class_output = output_path / f"class_{class_idx:02d}"
            generated = self.generate_images(
                class_idx=class_idx,
                num_images=images_per_class,
                output_dir=str(class_output),
                prefix="syn"
            )
            all_generated[class_idx] = generated

        return all_generated


def generate_synthetic_for_training(
    network_pkl: str,
    output_base_dir: str = "data/synthetic_images",
    images_per_class: int = 100,
    syndrome_mapping: Dict[str, int] = SYNDROME_CLASS_MAPPING
):
    """
    Generate synthetic images organized by syndrome folder for training.

    Args:
        network_pkl: Path to StyleGAN3 network pickle
        output_base_dir: Base output directory
        images_per_class: Number of images to generate per class
        syndrome_mapping: Mapping from syndrome name to class index
    """
    import PIL.Image

    output_path = Path(output_base_dir)

    print("Initializing generator...")
    generator = SyntheticImageGenerator(network_pkl)

    # Create directories for each syndrome
    for syndrome_name in syndrome_mapping.keys():
        (output_path / syndrome_name).mkdir(parents=True, exist_ok=True)

    # Generate images for each syndrome
    for syndrome_name, class_idx in syndrome_mapping.items():
        print(f"\nGenerating {images_per_class} images for {syndrome_name}...")

        syndrome_output = output_path / syndrome_name
        generator.generate_images(
            class_idx=class_idx,
            num_images=images_per_class,
            output_dir=str(syndrome_output),
            prefix=f"SYN_{syndrome_name[:5]}"
        )

    print(f"\nSynthetic dataset generated in {output_path}")

    # Print summary
    print("\nGenerated images per syndrome:")
    for syndrome_name in syndrome_mapping.keys():
        count = len(list((output_path / syndrome_name).glob("*.png")))
        print(f"  {syndrome_name}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic rare disease images")
    parser.add_argument("--network", type=str, required=True,
                        help="Path to StyleGAN3 pickle")
    parser.add_argument(
        "--outdir", type=str, default="data/synthetic_images", help="Output directory")
    parser.add_argument("--num", type=int, default=100,
                        help="Images per class")
    parser.add_argument("--trunc", type=float,
                        default=0.7, help="Truncation psi")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_synthetic_for_training(
        network_pkl=args.network,
        output_base_dir=args.outdir,
        images_per_class=args.num
    )


if __name__ == "__main__":
    main()
