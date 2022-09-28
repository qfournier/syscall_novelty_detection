import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
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
    ):
        """Concatenation of the system call embddings and encodings. To
        disable (ignore) an argument, set its dimension to 0.

        Args:
            n_syscall (int): number of distinct system call names.
            n_process (int): number of distinct process names.
            dim_sys (int): dimension of the system call name embedding.
            dim_entry (int): dimension of the entry/exit embedding.
            dim_ret (int): dimension of the return value embedding.
            dim_proc (int): dimension of the process name embedding.
            dim_pid (int): dimension of the PID encoding.
            dim_tid (int): dimension of the TID encoding.
            dim_order (int): dimension of the order encoding.
            dim_time (int): dimension of the encoding of the elapsed
            time between events.
        """
        super(Embedding, self).__init__()

        # Embeddings and encodings sizes
        self.dim_sys = dim_sys
        self.dim_entry = dim_entry
        self.dim_ret = dim_ret
        self.dim_proc = dim_proc
        self.dim_pid = dim_pid
        self.dim_tid = dim_tid
        self.dim_order = dim_order
        self.dim_time = dim_time

        # Create the necessary embedding layers
        self.emb_call = nn.Embedding(n_syscall, self.dim_sys, padding_idx=0)
        if self.dim_entry > 0:
            self.emb_entry = nn.Embedding(3, self.dim_entry, padding_idx=0)
        if self.dim_ret > 0:
            self.emb_ret = nn.Embedding(3, self.dim_ret, padding_idx=0)
        if self.dim_proc > 0:
            self.emb_proc = nn.Embedding(n_process,
                                         self.dim_proc,
                                         padding_idx=0)

        self.init_weights()

    def init_weights(self):
        """Initialize the classifier weights using the uniform
        distribution proposed by Xavier & Bengio.
        """
        nn.init.normal_(self.emb_call.weight, 0, self.dim_sys**-0.5)
        if self.dim_entry > 0:
            nn.init.normal_(self.emb_entry.weight, 0, self.dim_entry**-0.5)
        if self.dim_ret > 0:
            nn.init.normal_(self.emb_ret.weight, 0, self.dim_ret**-0.5)
        if self.dim_proc > 0:
            nn.init.normal_(self.emb_proc.weight, 0, self.dim_proc**-0.5)

    def forward(self, call, entry, ret, time, proc, pid, tid):
        """Compute the embedding of a batch of B sequence of length N.

        Args:
            call (Tensor): system call names (B x N).
            entry (Tensor): entry/exit (B x N).
            ret (Tensor): return value (B x N).
            time (Tensor): elapsed time between events (B x N).
            proc (Tensor): process names (B x N).
            pid (Tensor): PID (B x N).
            tid (Tensor): TID (B x N).

        Returns:
            Tensor: embedding (B x N x E).
        """
        size = call.shape

        emb = self.emb_call(call)

        if self.dim_entry > 0:
            emb = torch.cat((emb, self.emb_entry(entry)), -1)

        if self.dim_ret > 0:
            emb = torch.cat((emb, self.emb_ret(ret)), -1)

        if self.dim_proc > 0:
            emb = torch.cat((emb, self.emb_proc(proc)), -1)

        if self.dim_pid > 0:
            pid = pid.unsqueeze(2)
            denominator = torch.exp(
                torch.arange(0, self.dim_pid, 2).float() *
                (-math.log(1e6) / self.dim_pid)).to(call.device)
            pid_enc = torch.zeros(size[0], size[1],
                                  self.dim_pid).to(call.device)
            pid_enc[:, :, 0::2] = torch.sin(pid * denominator)
            pid_enc[:, :, 1::2] = torch.cos(pid * denominator)
            emb = torch.cat((emb, pid_enc), -1)

        if self.dim_tid > 0:
            tid = tid.unsqueeze(2)
            denominator = torch.exp(
                torch.arange(0, self.dim_tid, 2).float() *
                (-math.log(1e6) / self.dim_tid)).to(call.device)
            tid_enc = torch.zeros(size[0], size[1],
                                  self.dim_tid).to(call.device)
            tid_enc[:, :, 0::2] = torch.sin(tid * denominator)
            tid_enc[:, :, 1::2] = torch.cos(tid * denominator)
            emb = torch.cat((emb, tid_enc), -1)

        if self.dim_order > 0:
            ordering = (torch.arange(
                0, size[1], dtype=torch.float).unsqueeze(1).to(call.device))
            denominator = torch.exp(
                torch.arange(0, self.dim_order, 2).float() *
                (-math.log(1e6) / self.dim_order)).to(call.device)
            pos_enc = torch.zeros(size[0], size[1],
                                  self.dim_order).to(call.device)
            pos_enc[:, :, 0::2] = torch.sin(ordering * denominator)
            pos_enc[:, :, 1::2] = torch.cos(ordering * denominator)
            emb = torch.cat((emb, pos_enc), -1)

        if self.dim_time > 0:
            position = time.type(torch.float64).unsqueeze(2)
            denominator = torch.exp(
                torch.arange(0, self.dim_time, 2).float() *
                (-math.log(1e6) / self.dim_time)).to(call.device)
            pos_enc = torch.zeros(size[0], size[1],
                                  self.dim_time).to(call.device)
            pos_enc[:, :, 0::2] = torch.sin(position * denominator)
            pos_enc[:, :, 1::2] = torch.cos(position * denominator)
            emb = torch.cat((emb, pos_enc), -1)

        return emb
