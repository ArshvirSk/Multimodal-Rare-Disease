"""
Multimodal Fusion Module.
Implements concatenation, attention-based, and gated fusion strategies.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import get_config, FusionConfig


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion (baseline).

    Concatenates image and text embeddings and projects to hidden dimension.
    Z = W * [Z_image; Z_text] + b
    """

    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        """
        Initialize concatenation fusion.

        Args:
            image_dim: Image embedding dimension
            text_dim: Text embedding dimension
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, image_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_embedding: [batch_size, image_dim]
            text_embedding: [batch_size, text_dim]

        Returns:
            Fused embedding [batch_size, hidden_dim]
        """
        # Concatenate
        combined = torch.cat([image_embedding, text_embedding], dim=-1)

        # Project
        fused = self.fusion(combined)

        return fused


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Allows one modality to attend to the other.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-modal attention.

        Args:
            query_dim: Query dimension
            key_dim: Key/Value dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Projections
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: Query tensor [batch_size, query_dim]
            key: Key tensor [batch_size, key_dim]
            value: Value tensor [batch_size, key_dim]
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)

        # Add sequence dimension if not present
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)

        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Project
        Q = self.query_proj(query)  # [batch, seq_q, hidden]
        K = self.key_proj(key)  # [batch, seq_k, hidden]
        V = self.value_proj(value)  # [batch, seq_k, hidden]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape back
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.hidden_dim)
        )
        output = self.output_proj(output)

        # Squeeze sequence dimension if it was added
        if seq_len_q == 1:
            output = output.squeeze(1)

        return output, attention_weights


class AttentionFusion(nn.Module):
    """
    Attention-based multimodal fusion.

    Uses cross-modal attention to allow each modality to attend to the other,
    then combines the attended representations.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize attention fusion.

        Args:
            config: Fusion configuration. Uses default if None.
        """
        super().__init__()

        if config is None:
            config = get_config().fusion

        self.config = config
        self.hidden_dim = config.hidden_dim

        # Project modalities to same dimension
        self.image_proj = nn.Linear(config.image_proj_dim, config.hidden_dim)
        self.text_proj = nn.Linear(config.text_proj_dim, config.hidden_dim)

        # Cross-modal attention: image attends to text
        self.image_to_text_attention = CrossModalAttention(
            query_dim=config.hidden_dim,
            key_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # Cross-modal attention: text attends to image
        self.text_to_image_attention = CrossModalAttention(
            query_dim=config.hidden_dim,
            key_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # Layer normalization
        self.layer_norm_image = nn.LayerNorm(config.hidden_dim)
        self.layer_norm_text = nn.LayerNorm(config.hidden_dim)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Residual connection flag
        self.use_residual = config.use_residual

    def forward(
        self, image_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            image_embedding: [batch_size, image_dim]
            text_embedding: [batch_size, text_dim]

        Returns:
            Tuple of (fused_embedding, attention_info)
        """
        # Project to common dimension
        image_proj = self.image_proj(image_embedding)
        text_proj = self.text_proj(text_embedding)

        # Cross-modal attention
        # Image attends to text
        image_attended, image_attn_weights = self.image_to_text_attention(
            query=image_proj, key=text_proj, value=text_proj
        )

        # Text attends to image
        text_attended, text_attn_weights = self.text_to_image_attention(
            query=text_proj, key=image_proj, value=image_proj
        )

        # Residual connections and layer norm
        if self.use_residual:
            image_out = self.layer_norm_image(image_proj + image_attended)
            text_out = self.layer_norm_text(text_proj + text_attended)
        else:
            image_out = self.layer_norm_image(image_attended)
            text_out = self.layer_norm_text(text_attended)

        # Concatenate and fuse
        combined = torch.cat([image_out, text_out], dim=-1)
        fused = self.fusion(combined)

        # Return attention info for visualization
        attention_info = {
            "image_to_text_attention": image_attn_weights,
            "text_to_image_attention": text_attn_weights,
        }

        return fused, attention_info


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism.

    Learns gates to control the contribution of each modality.
    """

    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        """
        Initialize gated fusion.

        Args:
            image_dim: Image embedding dimension
            text_dim: Text embedding dimension
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project to common dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Gating mechanism
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self, image_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_embedding: [batch_size, image_dim]
            text_embedding: [batch_size, text_dim]

        Returns:
            Fused embedding [batch_size, hidden_dim]
        """
        # Project
        image_proj = self.image_proj(image_embedding)
        text_proj = self.text_proj(text_embedding)

        # Compute gate
        combined = torch.cat([image_proj, text_proj], dim=-1)
        gate = self.gate(combined)

        # Apply gate
        fused = gate * image_proj + (1 - gate) * text_proj

        # Output projection
        output = self.output(fused)

        return output


class MultimodalFusion(nn.Module):
    """
    Main fusion module that supports multiple fusion strategies.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize multimodal fusion.

        Args:
            config: Fusion configuration. Uses default if None.
        """
        super().__init__()

        if config is None:
            config = get_config().fusion

        self.config = config
        self.fusion_type = config.fusion_type

        if config.fusion_type == "concatenation":
            self.fusion_layer = ConcatenationFusion(
                image_dim=config.image_proj_dim,
                text_dim=config.text_proj_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        elif config.fusion_type == "attention":
            self.fusion_layer = AttentionFusion(config)
        elif config.fusion_type == "gated":
            self.fusion_layer = GatedFusion(
                image_dim=config.image_proj_dim,
                text_dim=config.text_proj_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion_type}")

    def forward(
        self, image_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass.

        Args:
            image_embedding: [batch_size, image_dim]
            text_embedding: [batch_size, text_dim]

        Returns:
            Tuple of (fused_embedding, attention_info or None)
        """
        if self.fusion_type == "attention":
            return self.fusion_layer(image_embedding, text_embedding)
        else:
            fused = self.fusion_layer(image_embedding, text_embedding)
            return fused, None


def create_fusion_module(
    fusion_type: str = "attention",
    image_dim: int = 512,
    text_dim: int = 768,
    hidden_dim: int = 512,
    **kwargs,
) -> MultimodalFusion:
    """
    Factory function to create fusion module.

    Args:
        fusion_type: Type of fusion (concatenation, attention, gated)
        image_dim: Image embedding dimension
        text_dim: Text embedding dimension
        hidden_dim: Output hidden dimension
        **kwargs: Additional configuration options

    Returns:
        Initialized fusion module
    """
    config = FusionConfig(
        fusion_type=fusion_type,
        image_proj_dim=image_dim,
        text_proj_dim=text_dim,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    return MultimodalFusion(config)


if __name__ == "__main__":
    # Test fusion modules
    print("Testing Fusion Modules...")

    batch_size = 4
    image_dim = 512
    text_dim = 768
    hidden_dim = 512

    # Create dummy embeddings
    image_emb = torch.randn(batch_size, image_dim)
    text_emb = torch.randn(batch_size, text_dim)

    print(f"Image embedding shape: {image_emb.shape}")
    print(f"Text embedding shape: {text_emb.shape}")

    # Test concatenation fusion
    print("\n--- Concatenation Fusion ---")
    concat_fusion = ConcatenationFusion(image_dim, text_dim, hidden_dim)
    output = concat_fusion(image_emb, text_emb)
    print(f"Output shape: {output.shape}")

    # Test attention fusion
    print("\n--- Attention Fusion ---")
    attention_fusion = create_fusion_module(
        fusion_type="attention",
        image_dim=image_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
    )
    output, attn_info = attention_fusion(image_emb, text_emb)
    print(f"Output shape: {output.shape}")
    print(f"Attention info keys: {attn_info.keys() if attn_info else 'None'}")

    # Test gated fusion
    print("\n--- Gated Fusion ---")
    gated_fusion = create_fusion_module(
        fusion_type="gated",
        image_dim=image_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
    )
    output, _ = gated_fusion(image_emb, text_emb)
    print(f"Output shape: {output.shape}")

    # Count parameters
    print("\n--- Parameter Counts ---")
    for name, module in [
        ("Concat", concat_fusion),
        ("Attention", attention_fusion),
        ("Gated", gated_fusion),
    ]:
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} parameters")
