"""
Multimodal Rare Disease Diagnosis Framework
Combines facial image analysis with clinical text for rare disease classification.
"""

__version__ = "1.1.0"
__author__ = "Multimodal ML Research"

from .config import Config, get_config
from .multimodal_classifier import MultimodalClassifier, ImageOnlyClassifier, TextOnlyClassifier
from .train_multimodal import train_multimodal, MultimodalTrainer
