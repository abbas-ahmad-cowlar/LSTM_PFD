"""
GPT (Generative Pre-trained Transformer) Implementation
Based on "Improving Language Understanding by Generative Pre-Training"
and "Language Models are Unsupervised Multitask Learners" (GPT-2)

This module provides a complete implementation of GPT architecture with support for:
- Causal (autoregressive) language modeling
- Various GPT configurations (GPT, GPT-2, GPT-3 style)
- Text generation with different sampling strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class GPTConfig:
    """Configuration class for GPT model."""
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation = activation
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon


class GPTAttention(nn.Module):
    """
    Multi-head causal self-attention for GPT.

    Args:
        config: GPT configuration
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Key, Query, Value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask to ensure attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            use_cache: Whether to return key-value cache
            past_key_value: Past key-value cache

        Returns:
            attn_output: Output tensor of shape (batch_size, seq_len, n_embd)
            present_key_value: Optional key-value cache
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Cache past key-values for faster generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        seq_len = k.size(2)
        att = att.masked_fill(self.bias[:, :, :T, :seq_len] == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            att = att + attention_mask

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, present_key_value


class GPTMLP(nn.Module):
    """
    Feed-forward network for GPT.

    Args:
        config: GPT configuration
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.activation = F.gelu if config.activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    """
    A single GPT transformer block.

    Args:
        config: GPT configuration
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPTMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional attention mask
            use_cache: Whether to return key-value cache
            past_key_value: Past key-value cache

        Returns:
            output: Output tensor of shape (batch_size, seq_len, n_embd)
            present_key_value: Optional key-value cache
        """
        # Self-attention with pre-norm
        attn_output, present_key_value = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        x = x + attn_output

        # Feed-forward with pre-norm
        x = x + self.mlp(self.ln_2(x))

        return x, present_key_value


class GPTModel(nn.Module):
    """
    GPT Model for autoregressive language modeling.

    Args:
        config: GPT configuration

    Examples:
        # GPT-Small (similar to GPT-1)
        >>> config = GPTConfig(
        ...     vocab_size=50257, n_positions=512, n_embd=768,
        ...     n_layer=12, n_head=12
        ... )
        >>> model = GPTModel(config)

        # GPT-Medium (GPT-2 Medium)
        >>> config = GPTConfig(
        ...     vocab_size=50257, n_positions=1024, n_embd=1024,
        ...     n_layer=24, n_head=16
        ... )
        >>> model = GPTModel(config)

        # GPT-Large (GPT-2 Large)
        >>> config = GPTConfig(
        ...     vocab_size=50257, n_positions=1024, n_embd=1280,
        ...     n_layer=36, n_head=20
        ... )
        >>> model = GPTModel(config)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between input embeddings and output layer
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            past_key_values: Past key-value cache
            use_cache: Whether to return key-value cache
            labels: Labels for language modeling loss

        Returns:
            Dictionary with 'logits', optionally 'loss' and 'past_key_values'
        """
        batch_size, seq_len = input_ids.shape

        # Get position IDs
        if position_ids is None:
            if past_key_values is not None:
                # Use cached position
                position_ids = torch.arange(
                    past_key_values[0][0].size(2),
                    seq_len + past_key_values[0][0].size(2),
                    dtype=torch.long,
                    device=input_ids.device
                )
            else:
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)

        # Get embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        hidden_states = self.drop(token_embeddings + position_embeddings)

        # Process attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Forward through transformer blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            if use_cache:
                presents.append(present_key_value)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Prepare output
        output = {'logits': logits}

        if use_cache:
            output['past_key_values'] = tuple(presents)

        # Calculate loss if labels are provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            output['loss'] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs of shape (batch_size, max_length)
        """
        self.eval()

        for _ in range(max_length - input_ids.size(1)):
            # Get predictions for the last token
            outputs = self(input_ids)
            logits = outputs['logits'][:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we exceed max position embeddings
            if input_ids.size(1) >= self.config.n_positions:
                break

        return input_ids


def gpt_small(vocab_size: int = 50257, **kwargs) -> GPTModel:
    """GPT-Small configuration (similar to GPT-1)."""
    config = GPTConfig(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        **kwargs
    )
    return GPTModel(config)


def gpt2_small(vocab_size: int = 50257, **kwargs) -> GPTModel:
    """GPT-2 Small (117M parameters)."""
    config = GPTConfig(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        **kwargs
    )
    return GPTModel(config)


def gpt2_medium(vocab_size: int = 50257, **kwargs) -> GPTModel:
    """GPT-2 Medium (345M parameters)."""
    config = GPTConfig(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        **kwargs
    )
    return GPTModel(config)


def gpt2_large(vocab_size: int = 50257, **kwargs) -> GPTModel:
    """GPT-2 Large (774M parameters)."""
    config = GPTConfig(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=1280,
        n_layer=36,
        n_head=20,
        **kwargs
    )
    return GPTModel(config)


def gpt2_xl(vocab_size: int = 50257, **kwargs) -> GPTModel:
    """GPT-2 XL (1.5B parameters)."""
    config = GPTConfig(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=1600,
        n_layer=48,
        n_head=25,
        **kwargs
    )
    return GPTModel(config)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create GPT-2 Small model
    model = gpt2_small().to(device)

    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)

    outputs = model(input_ids)
    logits = outputs['logits']
    print(f"Logits shape: {logits.shape}")  # (4, 128, 50257)

    # Test generation
    prompt = torch.randint(0, 50257, (1, 10)).to(device)
    generated = model.generate(
        prompt,
        max_length=50,
        temperature=0.8,
        top_k=40,
        do_sample=True
    )
    print(f"Generated shape: {generated.shape}")  # (1, 50)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
