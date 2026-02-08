"""
Text Encoder for Clinical Narrative Feature Extraction.
Supports BioBERT and ClinicalBERT models from HuggingFace.
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .config import get_config, TextEncoderConfig


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder for clinical narratives.

    Supports:
    - BioBERT (default): dmis-lab/biobert-base-cased-v1.2
    - ClinicalBERT: emilyalsentzer/Bio_ClinicalBERT
    - PubMedBERT: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract

    Output: 768-dimensional embedding vector (BERT hidden size)
    """

    def __init__(self, config: Optional[TextEncoderConfig] = None):
        """
        Initialize text encoder.

        Args:
            config: Text encoder configuration. Uses default if None.
        """
        super().__init__()

        if config is None:
            config = get_config().text_encoder

        self.config = config
        self.model_name = config.model_name
        self.embedding_dim = config.embedding_dim
        self.max_length = config.max_length
        self.use_pooler_output = config.use_pooler_output

        # Load pretrained model
        print(f"Loading text encoder: {self.model_name}")
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)

        # Verify embedding dimension matches
        actual_hidden_size = self.model_config.hidden_size
        if actual_hidden_size != self.embedding_dim:
            print(
                f"Note: Model hidden size ({actual_hidden_size}) differs from config ({self.embedding_dim})"
            )
            self.embedding_dim = actual_hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)

        # Optional projection layer
        self.projection = None

        # Freeze layers if specified
        if config.freeze_embeddings:
            self._freeze_embeddings()
        if config.freeze_layers > 0:
            self._freeze_layers(config.freeze_layers)

    def _freeze_embeddings(self):
        """Freeze the embedding layer."""
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        print("Froze embedding layer")

    def _freeze_layers(self, num_layers: int):
        """
        Freeze the first N transformer layers.

        Args:
            num_layers: Number of layers to freeze
        """
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        else:
            print("Warning: Could not find encoder layers to freeze")
            return

        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        print(f"Froze first {num_layers} transformer layers")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Text embedding [batch_size, embedding_dim]
        """
        # Get transformer outputs
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        if self.use_pooler_output and hasattr(outputs, "pooler_output"):
            # Use the pooler output (processed [CLS] token)
            embedding = outputs.pooler_output
        else:
            # Use the [CLS] token from the last hidden state
            embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        embedding = self.dropout(embedding)

        # Optional projection
        if self.projection is not None:
            embedding = self.projection(embedding)

        return embedding

    def get_all_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all hidden states for attention visualization.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (last_hidden_state, all_hidden_states)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs.last_hidden_state, outputs.hidden_states

    def get_attention_weights(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for visualization.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (embedding, attention_weights)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

        # Get embedding
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = self.dropout(embedding)

        # Get attention weights from all layers
        # attentions is a tuple of (batch, heads, seq_len, seq_len) tensors
        attention_weights = outputs.attentions

        return embedding, attention_weights


class BioBERTEncoder(TextEncoder):
    """Convenience class for BioBERT encoder."""

    def __init__(
        self,
        embedding_dim: int = 768,
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ):
        config = TextEncoderConfig(
            model_name="dmis-lab/biobert-base-cased-v1.2",
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )
        super().__init__(config)


class ClinicalBERTEncoder(TextEncoder):
    """Convenience class for ClinicalBERT encoder."""

    def __init__(
        self,
        embedding_dim: int = 768,
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ):
        config = TextEncoderConfig(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )
        super().__init__(config)


class PubMedBERTEncoder(TextEncoder):
    """Convenience class for PubMedBERT encoder."""

    def __init__(
        self,
        embedding_dim: int = 768,
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ):
        config = TextEncoderConfig(
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )
        super().__init__(config)


class TextEncoderWithProjection(TextEncoder):
    """
    Text encoder with additional projection layer.

    Useful when you need to match the CNN encoder embedding dimension.
    """

    def __init__(
        self, config: Optional[TextEncoderConfig] = None, output_dim: int = 512
    ):
        """
        Initialize text encoder with projection.

        Args:
            config: Text encoder configuration
            output_dim: Output embedding dimension after projection
        """
        super().__init__(config)

        # Add projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(output_dim, output_dim),
        )

        self.embedding_dim = output_dim


def create_text_encoder(
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    output_dim: Optional[int] = None,
    **kwargs,
) -> TextEncoder:
    """
    Factory function to create text encoder.

    Args:
        model_name: HuggingFace model name
        output_dim: If specified, add projection layer to this dimension
        **kwargs: Additional configuration options

    Returns:
        Initialized text encoder
    """
    config = TextEncoderConfig(model_name=model_name, **kwargs)

    if output_dim is not None:
        return TextEncoderWithProjection(config, output_dim=output_dim)

    return TextEncoder(config)


def get_tokenizer(
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
) -> AutoTokenizer:
    """
    Get the tokenizer for a text encoder model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Initialized tokenizer
    """
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # Test the text encoder
    print("Testing Text Encoder...")

    # Test BioBERT
    print("\n--- BioBERT Encoder ---")
    try:
        biobert_encoder = BioBERTEncoder(embedding_dim=768)
        print(f"Model: BioBERT")
        print(f"Output dim: {biobert_encoder.embedding_dim}")

        # Test forward pass
        tokenizer = get_tokenizer("dmis-lab/biobert-base-cased-v1.2")

        sample_text = [
            "Patient presents with hypertelorism, seizures, and delayed speech.",
            "Clinical features include facial dysmorphism and intellectual disability.",
        ]

        encoding = tokenizer(
            sample_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output = biobert_encoder(
            input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]
        )

        print(f"Input shape: {encoding['input_ids'].shape}")
        print(f"Output shape: {output.shape}")

        # Count parameters
        num_params = sum(p.numel() for p in biobert_encoder.parameters())
        print(f"Parameters: {num_params:,}")

    except Exception as e:
        print(f"Error testing BioBERT: {e}")
        print("Note: You may need to install transformers and download the model first")

    # Test encoder with projection
    print("\n--- BioBERT with Projection (512-d output) ---")
    try:
        projected_encoder = create_text_encoder(
            model_name="dmis-lab/biobert-base-cased-v1.2", output_dim=512
        )
        print(f"Output dim: {projected_encoder.embedding_dim}")

        output = projected_encoder(
            input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]
        )
        print(f"Output shape: {output.shape}")

    except Exception as e:
        print(f"Error: {e}")
