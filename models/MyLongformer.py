from .SwiGLU import swiglu

from .Embedding import Embedding
from .MyLongformerSelfAttention import MyLongformerSelfAttention

import torch
from torch import Tensor
from torch.nn.functional import relu
from torch.nn.functional import gelu
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.utils.checkpoint import checkpoint
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


# https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/
# modules/transformer.py
class MyLongformer(Module):
    def __init__(
        self,
        n_syscall,
        n_process,
        n_head,
        n_hidden,
        n_layer,
        dropout,
        dim_sys,
        dim_entry,
        dim_ret,
        dim_proc,
        dim_pid,
        dim_tid,
        dim_order,
        dim_time,
        activation,
        tfixup,
        window,
        dilatation,
        global_att,
    ):
        super(MyLongformer, self).__init__()

        # Dropout
        dropout = 0 if dropout is None else dropout

        # Embeddings and encodings sizes
        self.dim_sys = dim_sys
        self.dim_entry = dim_entry
        self.dim_ret = dim_ret
        self.dim_proc = dim_proc
        self.dim_pid = dim_pid
        self.dim_tid = dim_tid
        self.dim_order = dim_order
        self.dim_time = dim_time

        # Input embedding dimension
        self.d_model = sum(
            (
                dim_sys,
                dim_entry,
                dim_ret,
                dim_proc,
                dim_pid,
                dim_tid,
                dim_order,
                dim_time,
            )
        )
        # Number of heads and layers
        self.nhead = n_head
        self.nlayer = n_layer

        self.embedding = Embedding(
            n_syscall,
            n_process,
            dim_sys,
            dim_entry,
            dim_ret,
            dim_proc,
            dim_pid,
            dim_tid,
            dim_order,
            dim_time,
        )
        self.emb_dropout = Dropout(dropout)

        self.attention_window = window
        self.attention_dilation = dilatation
        self.global_att_idxs = global_att

        if tfixup:
            self.encoder = TransformerEncoder(
                n_layer,
                self.d_model,
                n_head,
                n_hidden,
                dropout,
                activation,
                attention_window=self.attention_window,
                attention_dilation=self.attention_dilation,
            )
        else:
            encoder_norm = LayerNorm(self.d_model)
            self.encoder = TransformerEncoder(
                n_layer,
                self.d_model,
                n_head,
                n_hidden,
                dropout,
                activation,
                norm=encoder_norm,
                attention_window=self.attention_window,
                attention_dilation=self.attention_dilation,
            )

        self.classifier = torch.nn.Linear(self.d_model, n_syscall)

        if tfixup:
            self._reset_parameters_tfixup()
        else:
            self._reset_parameters()

    def forward(
        self,
        call,
        entry,
        time,
        proc,
        pid,
        tid,
        ret,
        pad_mask,
        chk,
        save_attn=False,
    ):
        src = self.embedding(call, entry, ret, time, proc, pid, tid)
        src = self.emb_dropout(src)
        src = src.transpose(1, 0).contiguous()

        att_mask = self._generate_longformer_mask(
            src.size(1),
            src.size(0),
            self.global_att_idxs,
            pad_mask.to(src.device),
        ).to(src.device)

        memory = self.encoder(
            src, mask=att_mask, chk=chk, save_attn=save_attn,
        )

        memory = memory.transpose(1, 0).contiguous()
        output = self.classifier(memory)
        return output

    def _generate_longformer_mask(
        self, batch_sz, sz: int, global_indexes, pad_mask
    ):
        mask = torch.zeros(batch_sz, sz).to(pad_mask.device)
        mask[:, global_indexes] = 10000.0
        mask = mask.masked_fill(pad_mask, -10000.0)
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _reset_parameters_tfixup(self):
        with torch.no_grad():

            xavier_uniform_(self.classifier.weight)

            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

            for layer in self.encoder.layers:
                # Scale the MLP weights
                layer.linear1.weight *= 0.67 * (self.nlayer ** -0.25)
                layer.linear2.weight *= 0.67 * (self.nlayer ** -0.25)
                # Scale Wq, Wk, and Wv instead of just Wv because they are
                # grouped layer.self_attn.in_proj_weight *= 0.67 *
                # (self.nlayer ** -0.25)
                # For window attention
                layer.self_attn.query.weight *= 0.67 * (self.nlayer ** -0.25)
                layer.self_attn.key.weight *= 0.67 * (self.nlayer ** -0.25)
                layer.self_attn.value.weight *= 0.67 * (self.nlayer ** -0.25)
                # For global attention
                layer.self_attn.query.weight *= 0.67 * (self.nlayer ** -0.25)
                layer.self_attn.key.weight *= 0.67 * (self.nlayer ** -0.25)
                layer.self_attn.value.weight *= 0.67 * (self.nlayer ** -0.25)
                # Scale Wo
                layer.self_attn.out_proj.weight *= 0.67 * (
                    self.nlayer ** -0.25
                )

            self.embedding.emb_call.weight *= 9 * (self.nlayer ** -0.25)
            if self.dim_entry > 0:
                self.embedding.emb_entry.weight *= 9 * (self.nlayer ** -0.25)
            if self.dim_ret > 0:
                self.embedding.emb_ret.weight *= 9 * (self.nlayer ** -0.25)
            if self.dim_proc > 0:
                self.embedding.emb_proc.weight *= 9 * (self.nlayer ** -0.25)


class TransformerEncoder(Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        num_layers,
        d_model,
        n_head,
        dim_feedforward,
        dropout,
        activation,
        norm=None,
        attention_window=None,
        attention_dilation=None,
    ):
        super(TransformerEncoder, self).__init__()

        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_head,
                    i,
                    dim_feedforward,
                    dropout,
                    activation,
                    attention_window,
                    attention_dilation,
                )
                for i in range(num_layers)
            ]
        )

        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, src: Tensor, mask=None, chk=False, save_attn=False,
    ):

        output = src

        for layer in self.layers:
            if chk:
                # Cannot save the attention while using checkpoint
                output = checkpoint(layer, output, mask, False)
            else:
                output = layer(output, mask, save_attn)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model,
        nhead,
        layer_id,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        attention_window=None,
        attention_dilation=None,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MyLongformerSelfAttention(
            d_model,
            nhead,
            dropout,
            layer_id,
            attention_window,
            attention_dilation,
        )

        # Implementation of Feedforward model
        if activation == "swiglu":
            dim_feedforward_out = int(dim_feedforward * 2 / 3)
            dim_feedforward_in = 2 * dim_feedforward_out
        else:
            dim_feedforward_out = dim_feedforward
            dim_feedforward_in = dim_feedforward
        self.linear1 = Linear(d_model, dim_feedforward_in)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward_out, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = relu
        elif activation == "gelu":
            self.activation = gelu
        elif activation == "swiglu":
            self.activation = swiglu

    def forward(
        self, src: Tensor, attention_mask=None, save_attn=False,
    ):
        src2 = self.self_attn(
            src, attention_mask=attention_mask, save_attn=save_attn,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
