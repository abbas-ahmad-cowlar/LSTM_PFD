"""
T5 (Text-to-Text Transfer Transformer) Implementation
Based on "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
arXiv:1910.10683

This module provides a complete implementation of T5 architecture with support for:
- Encoder-decoder architecture with relative positional embeddings
- Various T5 configurations (Small, Base, Large, XL, XXL)
- Text generation and sequence-to-sequence tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class T5Config:
    """Configuration class for T5 model."""
    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_decoder_layers: Optional[int] = None,
        num_heads: int = 8,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        feed_forward_proj: str = "relu"
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.feed_forward_proj = feed_forward_proj


class T5LayerNorm(nn.Module):
    """
    T5 uses RMSNorm instead of standard LayerNorm.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # T5 uses root mean square layer normalization
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class T5Attention(nn.Module):
    """
    Multi-head attention with relative position embeddings for T5.

    Args:
        config: T5 configuration
        has_relative_attention_bias: Whether to compute relative attention bias
    """
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = False
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.num_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.num_heads * self.d_kv

        # Q, K, V projections
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.num_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """
        Compute relative position bucket for relative attention bias.

        Args:
            relative_position: Relative position tensor
            bidirectional: Whether attention is bidirectional
            num_buckets: Number of buckets
            max_distance: Maximum distance

        Returns:
            Bucketed relative positions
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # Now relative_position is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Use logarithmic bucketing for larger distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Compute relative attention bias."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: Input hidden states
            mask: Attention mask
            key_value_states: Key-value states for cross-attention
            position_bias: Position bias
            past_key_value: Past key-value cache
            use_cache: Whether to return cache

        Returns:
            attn_output: Attention output
            position_bias: Position bias
            present_key_value: Key-value cache
        """
        batch_size, seq_length = hidden_states.shape[:2]

        # Determine if this is self-attention or cross-attention
        real_seq_length = seq_length
        if past_key_value is not None:
            real_seq_length += past_key_value[0].shape[2]

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        # Compute Q, K, V
        def shape(states):
            return states.view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)

        def unshape(states):
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        query_states = shape(self.q(hidden_states))

        if key_value_states is None:
            # Self-attention
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
        else:
            # Cross-attention
            key_states = shape(self.k(key_value_states))
            value_states = shape(self.v(key_value_states))

        # Use cached key-value if available
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # Compute position bias
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(seq_length, key_length, device=scores.device)
            else:
                position_bias = torch.zeros(
                    (1, self.num_heads, seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype
                )

        scores += position_bias

        # Apply mask
        if mask is not None:
            scores = scores + mask

        # Normalize and apply attention
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = unshape(attn_output)

        # Output projection
        attn_output = self.o(attn_output)

        return attn_output, position_bias, present_key_value


class T5LayerSelfAttention(nn.Module):
    """Self-attention layer for T5."""
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias, present_key_value = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias, present_key_value


class T5LayerCrossAttention(nn.Module):
    """Cross-attention layer for T5 decoder."""
    def __init__(self, config: T5Config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias, present_key_value = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias, present_key_value


class T5LayerFF(nn.Module):
    """Feed-forward layer for T5."""
    def __init__(self, config: T5Config):
        super().__init__()
        self.DenseReluDense = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.ReLU() if config.feed_forward_proj == "relu" else nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout_rate)
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed_hidden_states)
        hidden_states = hidden_states + ff_output
        return hidden_states


class T5Block(nn.Module):
    """A single T5 encoder or decoder block."""
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()

        # Self-attention
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias))

        # Cross-attention for decoder
        if is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        # Feed-forward
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        """Forward pass through T5 block."""
        # Self-attention
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, position_bias, present_key_value_self = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache
        )

        present_key_value = present_key_value_self

        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_past_key_value = past_key_value[2:] if past_key_value is not None else None
            hidden_states, encoder_position_bias, present_key_value_cross = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_position_bias,
                past_key_value=cross_attn_past_key_value,
                use_cache=use_cache
            )
            if use_cache:
                present_key_value = present_key_value + present_key_value_cross

        # Feed-forward
        ff_layer_idx = 2 if self.is_decoder else 1
        hidden_states = self.layer[ff_layer_idx](hidden_states)

        return hidden_states, position_bias, encoder_position_bias, present_key_value


class T5Stack(nn.Module):
    """Stack of T5 encoder or decoder blocks."""
    def __init__(self, config: T5Config, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.num_layers = config.num_decoder_layers if is_decoder else config.num_layers

        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=(i == 0), is_decoder=is_decoder)
            for i in range(self.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        """Forward pass through T5 stack."""
        position_bias = None
        encoder_position_bias = None
        presents = () if use_cache else None

        for i, block in enumerate(self.block):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states, position_bias, encoder_position_bias, present_key_value = block(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_position_bias=encoder_position_bias,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

            if use_cache:
                presents = presents + (present_key_value,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states, presents


class T5Model(nn.Module):
    """
    T5 Model for sequence-to-sequence tasks.

    This implementation follows the original T5 paper and supports various
    configurations (T5-Small, T5-Base, T5-Large, T5-XL, T5-XXL).

    Args:
        config: T5 configuration

    Examples:
        # T5-Small
        >>> config = T5Config(
        ...     vocab_size=32128, d_model=512, d_kv=64, d_ff=2048,
        ...     num_layers=6, num_heads=8
        ... )
        >>> model = T5Model(config)

        # T5-Base
        >>> config = T5Config(
        ...     vocab_size=32128, d_model=768, d_kv=64, d_ff=3072,
        ...     num_layers=12, num_heads=12
        ... )
        >>> model = T5Model(config)

        # T5-Large
        >>> config = T5Config(
        ...     vocab_size=32128, d_model=1024, d_kv=64, d_ff=4096,
        ...     num_layers=24, num_heads=16
        ... )
        >>> model = T5Model(config)
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config

        # Shared embedding
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Encoder and decoder
        self.encoder = T5Stack(config, is_decoder=False)
        self.decoder = T5Stack(config, is_decoder=True)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.shared.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        factor = self.config.d_model ** -0.5
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Input token IDs for encoder
            decoder_input_ids: Input token IDs for decoder
            attention_mask: Attention mask for encoder
            decoder_attention_mask: Attention mask for decoder
            past_key_values: Past key-value cache
            use_cache: Whether to use cache
            labels: Labels for loss computation

        Returns:
            Dictionary with 'logits', optionally 'loss' and 'past_key_values'
        """
        # Encoder
        encoder_hidden_states = self.shared(input_ids)
        encoder_outputs, _ = self.encoder(
            encoder_hidden_states,
            attention_mask=attention_mask
        )

        # Decoder
        decoder_hidden_states = self.shared(decoder_input_ids)
        decoder_outputs, presents = self.decoder(
            decoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # LM head
        logits = self.lm_head(decoder_outputs)

        output = {'logits': logits}

        if use_cache:
            output['past_key_values'] = presents

        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            output['loss'] = loss

        return output


def t5_small(**kwargs) -> T5Model:
    """T5-Small configuration (60M parameters)."""
    config = T5Config(
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        **kwargs
    )
    return T5Model(config)


def t5_base(**kwargs) -> T5Model:
    """T5-Base configuration (220M parameters)."""
    config = T5Config(
        vocab_size=32128,
        d_model=768,
        d_kv=64,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        **kwargs
    )
    return T5Model(config)


def t5_large(**kwargs) -> T5Model:
    """T5-Large configuration (770M parameters)."""
    config = T5Config(
        vocab_size=32128,
        d_model=1024,
        d_kv=64,
        d_ff=4096,
        num_layers=24,
        num_heads=16,
        **kwargs
    )
    return T5Model(config)


def t5_xl(**kwargs) -> T5Model:
    """T5-XL configuration (3B parameters)."""
    config = T5Config(
        vocab_size=32128,
        d_model=2048,
        d_kv=128,
        d_ff=8192,
        num_layers=24,
        num_heads=32,
        **kwargs
    )
    return T5Model(config)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create T5-Small model
    model = t5_small().to(device)

    # Test forward pass
    batch_size = 4
    src_len = 64
    tgt_len = 32
    input_ids = torch.randint(0, 32128, (batch_size, src_len)).to(device)
    decoder_input_ids = torch.randint(0, 32128, (batch_size, tgt_len)).to(device)

    outputs = model(input_ids, decoder_input_ids)
    logits = outputs['logits']
    print(f"Logits shape: {logits.shape}")  # (4, 32, 32128)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
