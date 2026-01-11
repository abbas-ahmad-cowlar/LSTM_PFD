"""
BERT (Bidirectional Encoder Representations from Transformers) Implementation
Based on "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
arXiv:1810.04805

This module provides a complete implementation of BERT architecture with support for:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Various BERT configurations (Base, Large)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings: Token + Position + Segment Embeddings

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension size
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Number of token types (typically 2 for sentence A/B)
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            token_type_ids: Token type IDs of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine embeddings
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BERTSelfAttention(nn.Module):
    """
    Multi-head self-attention for BERT.

    Args:
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            context_layer: Output of shape (batch_size, seq_len, hidden_size)
            attention_probs: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        if attention_mask is not None:
            # Expand mask: (B, seq_len) -> (B, 1, 1, seq_len)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask

        # Normalize to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        return context_layer, attention_probs


class BERTSelfOutput(nn.Module):
    """Self-attention output projection with residual connection."""
    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    """Complete BERT attention layer with output projection."""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self = BERTSelfAttention(hidden_size, num_attention_heads, dropout)
        self.output = BERTSelfOutput(hidden_size, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs, attention_probs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output, attention_probs


class BERTIntermediate(nn.Module):
    """Feed-forward intermediate layer."""
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    """Feed-forward output layer with residual connection."""
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    """A single BERT encoder layer."""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = BERTAttention(hidden_size, num_attention_heads, dropout)
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(hidden_size, intermediate_size, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BERTEncoder(nn.Module):
    """Stack of BERT encoder layers."""
    def __init__(
        self,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_encoded_layers: bool = False
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            hidden_states: Input of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask
            output_all_encoded_layers: Whether to return outputs from all layers

        Returns:
            encoded_layers: List of layer outputs or just the last one
            all_attention_probs: List of attention weights from all layers
        """
        all_encoder_layers = []
        all_attention_probs = []

        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            all_attention_probs.append(attention_probs)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers, all_attention_probs


class BERTPooler(nn.Module):
    """Pooler to extract the [CLS] token representation."""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool the first token ([CLS] token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTModel(nn.Module):
    """
    BERT Model for generating contextualized representations.

    This is the base BERT model that outputs hidden states without any task-specific head.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Size of intermediate feed-forward layer
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Number of token types
        dropout: Dropout probability

    Examples:
        # BERT-Base
        >>> model = BERTModel(
        ...     vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        ...     num_attention_heads=12, intermediate_size=3072
        ... )

        # BERT-Large
        >>> model = BERTModel(
        ...     vocab_size=30522, hidden_size=1024, num_hidden_layers=24,
        ...     num_attention_heads=16, intermediate_size=4096
        ... )
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embeddings = BERTEmbeddings(
            vocab_size, hidden_size, max_position_embeddings,
            type_vocab_size, dropout
        )
        self.encoder = BERTEncoder(
            num_hidden_layers, hidden_size, num_attention_heads,
            intermediate_size, dropout
        )
        self.pooler = BERTPooler(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_all_encoded_layers: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            token_type_ids: Token type IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            output_all_encoded_layers: Whether to return outputs from all layers

        Returns:
            sequence_output: Last layer output of shape (batch_size, seq_len, hidden_size)
            pooled_output: Pooled [CLS] representation of shape (batch_size, hidden_size)
            all_attention_probs: List of attention weights from all layers
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)

        # Encode
        encoded_layers, all_attention_probs = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers
        )

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if output_all_encoded_layers:
            return encoded_layers, pooled_output, all_attention_probs
        else:
            return sequence_output, pooled_output, all_attention_probs


class BERTForMaskedLM(nn.Module):
    """BERT model with Masked Language Modeling head."""
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bert = BERTModel(
            vocab_size, hidden_size, num_hidden_layers,
            num_attention_heads, intermediate_size,
            max_position_embeddings, type_vocab_size, dropout
        )

        # MLM head
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        masked_lm_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs
            attention_mask: Attention mask
            masked_lm_labels: Labels for masked tokens

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        sequence_output, _, _ = self.bert(
            input_ids, token_type_ids, attention_mask
        )

        prediction_scores = self.cls(sequence_output)

        output = {'logits': prediction_scores}

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                masked_lm_labels.view(-1)
            )
            output['loss'] = masked_lm_loss

        return output


class BERTForSequenceClassification(nn.Module):
    """BERT model for sequence classification tasks."""
    def __init__(
        self,
        num_labels: int = 2,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BERTModel(
            vocab_size, hidden_size, num_hidden_layers,
            num_attention_heads, intermediate_size,
            max_position_embeddings, type_vocab_size, dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs
            attention_mask: Attention mask
            labels: Classification labels

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        _, pooled_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = {'logits': logits}

        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output['loss'] = loss

        return output


def bert_base(vocab_size: int = 30522, **kwargs) -> BERTModel:
    """BERT-Base configuration."""
    return BERTModel(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        **kwargs
    )


def bert_large(vocab_size: int = 30522, **kwargs) -> BERTModel:
    """BERT-Large configuration."""
    return BERTModel(
        vocab_size=vocab_size,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create BERT-Base model
    model = bert_base().to(device)

    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)

    sequence_output, pooled_output, _ = model(
        input_ids, token_type_ids, attention_mask
    )

    print(f"Sequence output shape: {sequence_output.shape}")  # (4, 128, 768)
    print(f"Pooled output shape: {pooled_output.shape}")  # (4, 768)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
