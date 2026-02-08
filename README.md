# Multimodal Machine Learning Framework for Rare Genetic Disease Diagnosis

A deep learning framework that combines **facial phenotype analysis** and **clinical narrative understanding** for automated rare genetic disease classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This framework implements a multimodal AI diagnostic system that fuses:

- **CNN-based facial encoder** (ResNet50/EfficientNet-B0) for facial dysmorphism detection
- **BioBERT/ClinicalBERT text encoder** for clinical narrative understanding
- **Attention-based fusion** for combining modalities
- **Statistical validation** using Chi-square tests

### Supported Syndromes

The model is trained to classify 10 rare genetic syndromes:

1. Cornelia de Lange Syndrome
2. Williams-Beuren Syndrome
3. Noonan Syndrome
4. Kabuki Syndrome
5. KBG Syndrome
6. Angelman Syndrome
7. Rubinstein-Taybi Syndrome
8. Smith-Magenis Syndrome
9. Nicolaides-Baraitser Syndrome
10. 22q11.2 Deletion Syndrome

## 📁 Project Structure

```
multimodal-rare-disease/
├── data/
│   ├── FGDD/                    # FGDD phenotype data (Figshare)
│   ├── orphadata/               # Orphadata XML files
│   │   ├── orphadata_diseases.xml
│   │   ├── orphadata_phenotypes.xml
│   │   └── orphadata_genes.xml
│   └── hpo/                     # Human Phenotype Ontology
│       ├── hp.obo
│       └── phenotype.hpoa
├── PDIDB/                       # Synthetic facial image generator
├── src/
│   ├── config.py                # Configuration and hyperparameters
│   ├── image_dataset_loader.py  # Image preprocessing + augmentation
│   ├── text_dataset_loader.py   # Clinical text processing
│   ├── cnn_encoder.py           # ResNet50/EfficientNet encoder
│   ├── text_encoder.py          # BioBERT/ClinicalBERT encoder
│   ├── fusion_model.py          # Attention-based fusion
│   ├── multimodal_classifier.py # Complete multimodal model
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation metrics
│   ├── predict.py               # Inference pipeline
│   └── chi_square_test.py       # Statistical validation
├── notebooks/
│   └── explainability.ipynb     # Grad-CAM + attention visualization
├── results/
├── checkpoints/
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/multimodal-rare-disease.git
cd multimodal-rare-disease

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Train the model (with data augmentation for small datasets)
python run_training.py --image-dirs data/images_augmented --epochs 60

# Make a prediction on a new image
python predict.py --image path/to/face.jpg
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone or navigate to the project
cd multimodal-rare-disease

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Models

The framework automatically downloads pretrained models on first run:

- ResNet50 (ImageNet weights)
- BioBERT (dmis-lab/biobert-base-cased-v1.2)

## 📊 Data Sources

### Facial Images (PDIDB)

Synthetic facial images generated using StyleGAN3, trained on GestaltMatcher Database.

Repository: https://github.com/WGLab/PDIDB

### Clinical Text (Orphadata + HPO)

- **Orphadata**: Rare disease descriptions and phenotype associations
- **HPO**: Human Phenotype Ontology standard vocabulary

## 🏋️ Training

### Quick Smoke Test

```bash
python -m src.train --mode multimodal --smoke_test
```

### Full Training

```bash
# Multimodal (image + text)
python -m src.train --mode multimodal --epochs 100

# Image-only baseline
python -m src.train --mode image_only --epochs 100

# Text-only baseline
python -m src.train --mode text_only --epochs 100
```

### Training Options

```
--mode          Training mode: multimodal, image_only, text_only
--epochs        Number of training epochs (default: 100)
--batch_size    Batch size (default: 16)
--lr            Learning rate (default: 1e-4)
--device        Device: cuda or cpu
```

## 📈 Evaluation

```bash
python -m src.evaluate --checkpoint checkpoints/multimodal_best.pt --mode multimodal
```

### Metrics Computed

- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Confusion matrix
- ROC-AUC curves

## 🔬 Statistical Validation

Compare multimodal vs unimodal using Chi-square test:

```bash
# Run demo with synthetic data
python -m src.chi_square_test --demo

# Run on real predictions
python -m src.chi_square_test --predictions_dir results
```

**Hypothesis Testing:**

- H0: Multimodal and unimodal have same performance
- H1: Multimodal outperforms unimodal (p < 0.05)

## 🔮 Inference

```bash
python -m src.predict \
    --image path/to/face.jpg \
    --text "Patient presents with hypertelorism, seizures, and delayed speech." \
    --checkpoint checkpoints/multimodal_best.pt \
    --output prediction.json
```

### Output Example

```json
{
  "predictions": [
    {
      "syndrome": "Angelman Syndrome",
      "confidence": 0.85,
      "probability_percent": 85.0
    },
    {
      "syndrome": "Williams-Beuren Syndrome",
      "confidence": 0.08,
      "probability_percent": 8.0
    }
  ],
  "top_prediction": {
    "syndrome": "Angelman Syndrome",
    "confidence": 0.85
  }
}
```

## 🧠 Model Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Facial Image   │     │ Clinical Text   │
│  (224×224×3)    │     │ (max 128 tokens)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  CNN Encoder    │     │  Text Encoder   │
│  (ResNet50)     │     │  (BioBERT)      │
│  → 512-d        │     │  → 768-d        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Attention Fusion     │
         │  Cross-modal Attention│
         │  → 512-d              │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Classification Head  │
         │  FC → ReLU → Dropout  │
         │  → N syndromes        │
         └───────────────────────┘
```

## 🎨 Explainability

The framework includes explainability features:

1. **Grad-CAM**: Visualize which facial regions influence predictions
2. **Attention Weights**: Understand which clinical terms are important
3. **Cross-modal Attention**: See how image and text modalities interact

See `notebooks/explainability.ipynb` for interactive visualization.

## 📋 Configuration

All hyperparameters are centralized in `src/config.py`:

```python
# Key configurations
config.cnn_encoder.backbone = "resnet50"  # or "efficientnet_b0"
config.text_encoder.model_name = "dmis-lab/biobert-base-cased-v1.2"
config.fusion.fusion_type = "attention"  # or "concatenation", "gated"
config.training.learning_rate = 1e-4
config.training.batch_size = 16
config.training.num_epochs = 100
```

## 📚 References

### Datasets

- [PDIDB - Phenotype Disease Image Database](https://github.com/WGLab/PDIDB)
- [FGDD - Facial Gestalt Disease Database](https://doi.org/10.6084/m9.figshare.28516604)
- [Orphadata](https://www.orphadata.com/)
- [Human Phenotype Ontology](https://hpo.jax.org/)

### Models

- [BioBERT](https://github.com/dmis-lab/biobert)
- [ResNet](https://arxiv.org/abs/1512.03385)

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This is a research tool for educational purposes. **Do not use for clinical diagnosis.**
All predictions should be validated by qualified medical professionals.
