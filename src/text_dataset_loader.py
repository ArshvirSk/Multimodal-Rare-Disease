"""
Text Dataset Loader for Clinical Narratives.
Handles Orphadata XML parsing, HPO term processing, and text tokenization.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .config import get_config


class OrphadataParser:
    """
    Parser for Orphadata XML files containing rare disease information.
    """

    def __init__(
        self,
        diseases_file: Path,
        phenotypes_file: Path,
        genes_file: Optional[Path] = None,
    ):
        """
        Initialize Orphadata parser.

        Args:
            diseases_file: Path to orphadata_diseases.xml
            phenotypes_file: Path to orphadata_phenotypes.xml
            genes_file: Path to orphadata_genes.xml (optional)
        """
        self.diseases_file = Path(diseases_file)
        self.phenotypes_file = Path(phenotypes_file)
        self.genes_file = Path(genes_file) if genes_file else None

        self.diseases = {}
        self.phenotypes = {}
        self.genes = {}

        self._parse_all()

    def _parse_all(self):
        """Parse all Orphadata files."""
        print("Parsing Orphadata files...")

        if self.diseases_file.exists():
            self._parse_diseases()
        else:
            print(f"Warning: Diseases file not found: {self.diseases_file}")

        if self.phenotypes_file.exists():
            self._parse_phenotypes()
        else:
            print(f"Warning: Phenotypes file not found: {self.phenotypes_file}")

        if self.genes_file and self.genes_file.exists():
            self._parse_genes()

    def _parse_diseases(self):
        """Parse disease definitions from Orphadata."""
        try:
            tree = ET.parse(self.diseases_file)
            root = tree.getroot()

            # Find all disorder elements
            for disorder in root.iter("Disorder"):
                orpha_code = None
                name = None
                definition = None

                # Get OrphaCode
                orpha_code_elem = disorder.find(".//OrphaCode")
                if orpha_code_elem is not None:
                    orpha_code = orpha_code_elem.text

                # Get Name
                name_elem = disorder.find(".//Name")
                if name_elem is not None:
                    name = name_elem.text

                # Get Definition/Summary
                summary_elem = disorder.find(".//SummaryInformation")
                if summary_elem is not None:
                    def_elem = summary_elem.find(".//Definition")
                    if def_elem is not None:
                        definition = def_elem.text

                if orpha_code and name:
                    self.diseases[orpha_code] = {
                        "name": name,
                        "definition": definition or "",
                        "phenotypes": [],
                        "genes": [],
                    }

            print(f"Parsed {len(self.diseases)} diseases from Orphadata")

        except Exception as e:
            print(f"Error parsing diseases file: {e}")

    def _parse_phenotypes(self):
        """Parse disease-phenotype associations."""
        try:
            tree = ET.parse(self.phenotypes_file)
            root = tree.getroot()

            for disorder in root.iter("Disorder"):
                orpha_code = None

                orpha_code_elem = disorder.find(".//OrphaCode")
                if orpha_code_elem is not None:
                    orpha_code = orpha_code_elem.text

                if not orpha_code:
                    continue

                phenotype_list = []

                # Find HPO associations
                for hpo_assoc in disorder.iter("HPODisorderAssociation"):
                    hpo_elem = hpo_assoc.find(".//HPO")
                    if hpo_elem is not None:
                        hpo_id = hpo_elem.find(".//HPOId")
                        hpo_term = hpo_elem.find(".//HPOTerm")

                        if hpo_id is not None and hpo_term is not None:
                            phenotype_list.append(
                                {"hpo_id": hpo_id.text, "term": hpo_term.text}
                            )

                self.phenotypes[orpha_code] = phenotype_list

            print(f"Parsed phenotypes for {len(self.phenotypes)} diseases")

        except Exception as e:
            print(f"Error parsing phenotypes file: {e}")

    def _parse_genes(self):
        """Parse disease-gene associations."""
        if not self.genes_file:
            return

        try:
            tree = ET.parse(self.genes_file)
            root = tree.getroot()

            for disorder in root.iter("Disorder"):
                orpha_code = None

                orpha_code_elem = disorder.find(".//OrphaCode")
                if orpha_code_elem is not None:
                    orpha_code = orpha_code_elem.text

                if not orpha_code:
                    continue

                gene_list = []

                for gene_assoc in disorder.iter("DisorderGeneAssociation"):
                    gene_elem = gene_assoc.find(".//Gene")
                    if gene_elem is not None:
                        gene_symbol = gene_elem.find(".//Symbol")
                        gene_name = gene_elem.find(".//Name")

                        if gene_symbol is not None:
                            gene_list.append(
                                {
                                    "symbol": gene_symbol.text,
                                    "name": (
                                        gene_name.text if gene_name is not None else ""
                                    ),
                                }
                            )

                self.genes[orpha_code] = gene_list

            print(f"Parsed gene associations for {len(self.genes)} diseases")

        except Exception as e:
            print(f"Error parsing genes file: {e}")

    def get_disease_narrative(self, orpha_code: str) -> str:
        """
        Generate a clinical narrative for a disease.

        Args:
            orpha_code: Orphadata disease code

        Returns:
            Clinical narrative text
        """
        if orpha_code not in self.diseases:
            return ""

        disease = self.diseases[orpha_code]
        narrative_parts = []

        # Disease name
        narrative_parts.append(f"Patient diagnosed with {disease['name']}.")

        # Definition
        if disease["definition"]:
            narrative_parts.append(disease["definition"])

        # Phenotypes
        phenotypes = self.phenotypes.get(orpha_code, [])
        if phenotypes:
            terms = [p["term"] for p in phenotypes[:10]]  # Limit to top 10
            phenotype_text = f"Clinical features include: {', '.join(terms)}."
            narrative_parts.append(phenotype_text)

        # Genes
        genes = self.genes.get(orpha_code, [])
        if genes:
            gene_symbols = [g["symbol"] for g in genes[:5]]  # Limit to top 5
            gene_text = f"Associated genes: {', '.join(gene_symbols)}."
            narrative_parts.append(gene_text)

        return " ".join(narrative_parts)

    def get_all_narratives(self) -> Dict[str, str]:
        """Get narratives for all diseases."""
        narratives = {}
        for orpha_code in self.diseases:
            narratives[orpha_code] = self.get_disease_narrative(orpha_code)
        return narratives


class HPOParser:
    """
    Parser for Human Phenotype Ontology (HPO) files.
    """

    def __init__(self, hpo_file: Path, annotations_file: Optional[Path] = None):
        """
        Initialize HPO parser.

        Args:
            hpo_file: Path to hp.obo file
            annotations_file: Path to phenotype.hpoa file (optional)
        """
        self.hpo_file = Path(hpo_file)
        self.annotations_file = Path(annotations_file) if annotations_file else None

        self.terms = {}  # HPO ID -> term info
        self.annotations = {}  # Disease -> HPO terms

        self._parse_obo()
        if self.annotations_file:
            self._parse_annotations()

    def _parse_obo(self):
        """Parse the HPO OBO file."""
        if not self.hpo_file.exists():
            print(f"Warning: HPO file not found: {self.hpo_file}")
            return

        try:
            current_term = None

            with open(self.hpo_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if line == "[Term]":
                        current_term = {}
                    elif line == "" and current_term:
                        if "id" in current_term:
                            self.terms[current_term["id"]] = current_term
                        current_term = None
                    elif current_term is not None:
                        if line.startswith("id:"):
                            current_term["id"] = line[4:].strip()
                        elif line.startswith("name:"):
                            current_term["name"] = line[6:].strip()
                        elif line.startswith("def:"):
                            # Extract definition text
                            match = re.search(r'"([^"]*)"', line)
                            if match:
                                current_term["definition"] = match.group(1)
                        elif line.startswith("is_a:"):
                            if "parents" not in current_term:
                                current_term["parents"] = []
                            parent_id = line[6:].split("!")[0].strip()
                            current_term["parents"].append(parent_id)

            print(f"Parsed {len(self.terms)} HPO terms")

        except Exception as e:
            print(f"Error parsing HPO file: {e}")

    def _parse_annotations(self):
        """Parse HPO annotations file."""
        if not self.annotations_file.exists():
            print(f"Warning: Annotations file not found: {self.annotations_file}")
            return

        try:
            with open(self.annotations_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        database_id = parts[0]
                        disease_name = parts[1]
                        hpo_id = parts[3]

                        key = f"{database_id}:{disease_name}"
                        if key not in self.annotations:
                            self.annotations[key] = []
                        self.annotations[key].append(hpo_id)

            print(f"Parsed annotations for {len(self.annotations)} diseases")

        except Exception as e:
            print(f"Error parsing annotations: {e}")

    def get_term_name(self, hpo_id: str) -> str:
        """Get the name for an HPO term."""
        if hpo_id in self.terms:
            return self.terms[hpo_id].get("name", "")
        return ""

    def generate_phenotype_text(self, hpo_ids: List[str]) -> str:
        """
        Generate natural language text from HPO IDs.

        Args:
            hpo_ids: List of HPO IDs

        Returns:
            Natural language phenotype description
        """
        terms = []
        for hpo_id in hpo_ids:
            name = self.get_term_name(hpo_id)
            if name:
                terms.append(name)

        if not terms:
            return ""

        return f"Patient presents with {', '.join(terms)}."


class ClinicalTextDataset(Dataset):
    """
    PyTorch Dataset for clinical text data.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.2",
        max_length: int = 128,
    ):
        """
        Initialize clinical text dataset.

        Args:
            texts: List of clinical narrative texts
            labels: List of syndrome labels
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum token sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # Initialize tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tokenized sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class MultimodalDataset(Dataset):
    """
    Combined dataset for multimodal learning with images and text.
    """

    def __init__(
        self,
        image_paths: List[Path],
        texts: List[str],
        labels: List[int],
        image_transform=None,
        tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.2",
        max_length: int = 128,
    ):
        """
        Initialize multimodal dataset.

        Args:
            image_paths: List of image file paths
            texts: List of clinical narrative texts
            labels: List of syndrome labels
            image_transform: Image transforms
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum token sequence length
        """
        assert (
            len(image_paths) == len(texts) == len(labels)
        ), "Image paths, texts, and labels must have same length"

        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_transform = image_transform
        self.max_length = max_length

        # Initialize tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a multimodal sample.

        Returns:
            Dictionary with image, input_ids, attention_mask, and label
        """
        from PIL import Image

        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), color="gray")

        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize text
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = self.labels[idx]

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_syndrome_text_mapping(
    orphadata_parser: OrphadataParser, syndrome_names: List[str]
) -> Dict[str, str]:
    """
    Create a mapping from syndrome names to their clinical narratives.

    Args:
        orphadata_parser: Initialized OrphadataParser
        syndrome_names: List of target syndrome names

    Returns:
        Dictionary mapping syndrome names to narratives
    """
    mapping = {}

    # Try to find matching diseases in Orphadata
    for syndrome in syndrome_names:
        best_match = None
        best_score = 0

        for orpha_code, disease in orphadata_parser.diseases.items():
            disease_name = disease["name"].lower()
            syndrome_lower = syndrome.lower()

            # Simple matching - can be improved with fuzzy matching
            if syndrome_lower in disease_name or disease_name in syndrome_lower:
                score = len(syndrome_lower)
                if score > best_score:
                    best_score = score
                    best_match = orpha_code

        if best_match:
            mapping[syndrome] = orphadata_parser.get_disease_narrative(best_match)
        else:
            # Generate a basic narrative if no match found
            mapping[syndrome] = (
                f"Patient diagnosed with {syndrome}. "
                f"This is a rare genetic disorder with characteristic features."
            )

    return mapping


def prepare_multimodal_data(
    image_dir: Path, syndrome_names: List[str], orphadata_parser: OrphadataParser
) -> Tuple[List[Path], List[str], List[int]]:
    """
    Prepare paired image-text data for multimodal training.

    Args:
        image_dir: Directory containing syndrome image folders
        syndrome_names: List of syndrome names
        orphadata_parser: Initialized OrphadataParser

    Returns:
        Tuple of (image_paths, texts, labels)
    """
    # Get text mapping
    text_mapping = create_syndrome_text_mapping(orphadata_parser, syndrome_names)

    image_paths = []
    texts = []
    labels = []

    syndrome_to_idx = {name: idx for idx, name in enumerate(syndrome_names)}

    for syndrome in syndrome_names:
        syndrome_dir = Path(image_dir) / syndrome

        if not syndrome_dir.exists():
            print(f"Warning: Syndrome directory not found: {syndrome_dir}")
            continue

        syndrome_text = text_mapping.get(syndrome, f"Patient with {syndrome}.")
        label = syndrome_to_idx[syndrome]

        for img_path in syndrome_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                image_paths.append(img_path)
                texts.append(syndrome_text)
                labels.append(label)

    print(f"Prepared {len(image_paths)} multimodal samples")
    return image_paths, texts, labels


if __name__ == "__main__":
    # Test the text dataset loader
    config = get_config()

    print("Testing Text Dataset Loader...")
    print(f"Orphadata diseases: {config.data.orphadata_diseases}")
    print(f"HPO ontology: {config.data.hpo_ontology}")

    # Test Orphadata parser
    if config.data.orphadata_diseases.exists():
        parser = OrphadataParser(
            diseases_file=config.data.orphadata_diseases,
            phenotypes_file=config.data.orphadata_phenotypes,
            genes_file=config.data.orphadata_genes,
        )
        print(f"Loaded {len(parser.diseases)} diseases")

    # Test HPO parser
    if config.data.hpo_ontology.exists():
        hpo = HPOParser(
            hpo_file=config.data.hpo_ontology,
            annotations_file=config.data.hpo_annotations,
        )
        print(f"Loaded {len(hpo.terms)} HPO terms")
