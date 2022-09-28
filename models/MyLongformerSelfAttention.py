import uuid
import torch

from longformer.longformer import LongformerSelfAttention
from longformer.longformer import LongformerConfig


class MyLongformerSelfAttention(LongformerSelfAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        layer_id,
        attention_window=None,
        attention_dilation=None,
    ):
        """
        Args:
            attention_window: list of attention window sizes of length = number
                of layers.
            window size = number of attention locations on each side.
            For an affective window size of 512, use `attention_window=[256] *
                num_layers` which is 256 on each side.
            attention_dilation: list of attention dilation of length = number
                of layers.
            attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of
                both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM
                implemenation of Longformer selfattention, 'sliding_chunks'
                for another implementation of Longformer selfattention
        """
        config = LongformerConfig()

        config.hidden_size = embed_dim
        config.num_attention_heads = num_heads
        config.attention_probs_dropout_prob = dropout

        config.attention_mode = "tvm"
        config.autoregressive = True

        config.attention_window = attention_window
        config.attention_dilation = attention_dilation

        super().__init__(config, layer_id)

        # Apparently, there is no final Linear layer in the original
        # LongSelfAttention implementation
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def _combine_attn_and_save(self, local_attn, global_attn, attention_mask):
        # Combine and average both tensors and put in (bsz x seq_len x seq_len)
        batch_size, n_heads, seq_len, window_size = local_attn.size()

        # Average wrt the heads
        local_attn = torch.mean(local_attn, 1)
        global_attn = torch.mean(global_attn, 1)

        global_indexes = attention_mask[0, :] > 0
        global_indexes = global_indexes.nonzero(as_tuple=True)

        full_matrix = torch.zeros(batch_size, seq_len, seq_len)

        for batch in range(batch_size):
            for token in range(seq_len):
                for i in range(window_size):
                    if token + i - window_size >= 0:
                        full_matrix[
                            batch, token, token + 1 + i - window_size
                        ] += local_attn[batch, token, i]

                for i, idx in enumerate(global_indexes):
                    full_matrix[batch, token, idx] += global_attn[
                        batch, i, token
                    ]

        # Save the full matrix now
        torch.save(
            full_matrix,
            f"/tmp/attn_output_weights/{self.layer_id}_{uuid.uuid4()}",
        )

    def forward(
        self, hidden_states, attention_mask=None, save_attn=False,
    ):
        # with autocast(enabled=False):
        output_long = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=save_attn,
        )

        # Last projection layer
        output = self.out_proj(output_long[0])

        # Save the attention activation
        if save_attn:
            local_attn = output_long[1].cpu()
            global_attn = output_long[2].cpu()

            self._combine_attn_and_save(
                local_attn, global_attn, attention_mask
            )

        return output
