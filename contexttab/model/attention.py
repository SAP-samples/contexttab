# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.roberta.modeling_roberta import RobertaIntermediate, RobertaOutput, RobertaSelfOutput


class TwoDimensionalAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        layer_class = TorchRobertaLayer
        self.cross_column_layer = layer_class(config)
        self.cross_row_layer = layer_class(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: shape (num_rows, num_columns, hidden_size)
        attention_mask: shape (num_rows, num_rows) and values 0 (attend) or -inf (do not attend).
        Applies the two separate RobertaLayers, once along the "columns" direction and once along the "rows" direction.
        Along the columns direction, the attention is always full.
        Along the rows direction, an attention mask should be provided (to avoid that context rows can attend query rows)

        Returns: tensor of shape (num_rows, num_columns, hidden_size)
        """
        num_rows, num_columns, _ = hidden_states.shape

        # Cross-column attention it's easy: just treat row as distinct elements in the batch, so nothing to reshape
        horizontal_outputs = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        max_rows_per_batch = 8192
        col_fraction = 100.0 / float(num_columns)
        batch_step = int(max_rows_per_batch * col_fraction)

        for i in range(0, num_rows, batch_step):
            end_idx = i + batch_step
            chunk = hidden_states[i:end_idx, :, :]
            chunk_output = self.cross_column_layer(chunk)[0]
            horizontal_outputs[i:i + batch_step] = chunk_output

        # horizontal_outputs has shape (num_rows, num_columns, hidden_size)

        # For cross-row attention we can do the same, but we need to permute before and after.
        # Also, we use the attention mask here, if provided. But the attention mask has shape
        # (num_rows, num_rows) and we we unsqueeze to shape (1, 1, num_rows, num_rows)
        # (first two dims are batch size, which here is num_columns, and num_heads)
        batch_step = 100
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        horizontal_outputs = horizontal_outputs.transpose(0, 1).contiguous()

        vertical_outputs = torch.zeros_like(horizontal_outputs,
                                            dtype=horizontal_outputs.dtype,
                                            device=horizontal_outputs.device)
        for i in range(0, num_columns, batch_step):
            end_idx = i + batch_step
            chunk = horizontal_outputs[i:end_idx, :, :]
            chunk_output = self.cross_row_layer(chunk, attention_mask)[0]
            vertical_outputs[i:i + batch_step, :, :] = chunk_output

        return vertical_outputs.transpose(0, 1).contiguous()


class TorchRobertaLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = TorchAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor]:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        # Return a tuple like the original RobertaLayer, though it's not very useful
        return (self.output(intermediate_output, attention_output),)


class TorchAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention = TorchSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor]:
        self_outputs = self.self_attention(hidden_states, attention_mask)
        return self.output(self_outputs, hidden_states)


class TorchSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                             f"heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0, f'{config.hidden_size=} must be divisible by {config.num_attention_heads=}'
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob

        if config.attention_implementation == 'efficient' and torch.cuda.is_available():
            self.backend = SDPBackend.EFFICIENT_ATTENTION
        elif config.attention_implementation == 'math' or not torch.cuda.is_available():
            self.backend = SDPBackend.MATH
        else:
            raise ValueError(f"Unknown attention implementation {config.attention_implementation}")

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        hidden_states: shape (batch_size, seq_len, hidden_size)
        attention_mask: shape (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        with sdpa_kernel(self.backend):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False)

        context_layer = attn_output.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.hidden_size,)
        return context_layer.view(new_context_layer_shape)
