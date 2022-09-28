import torch.nn as nn

from . import Embedding


class LSTM(nn.Module):
    def __init__(
        self,
        n_syscall,
        n_process,
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
    ):
        super(LSTM, self).__init__()

        # Dropout
        dropout = 0 if dropout is None else dropout

        # Compute the embedding size
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

        # Embedding
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

        self.emb_dropout = nn.Dropout(dropout)

        # LSTM
        self.hidden_dim = n_hidden
        self.layers = n_layer
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            batch_first=True,
            dropout=dropout,
        )

        # Classifier
        self.classifier = nn.Linear(n_hidden, n_syscall)

        self.init_weights()

    def init_weights(self):
        """Initialize the classifier weights using the uniform
        distribution proposed by Xavier & Bengio."""
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.zero_()

    def forward(self, call, entry, time, proc, pid, tid, ret, *args, **kwargs):
        src = self.embedding(call, entry, ret, time, proc, pid, tid)
        src = self.emb_dropout(src)
        h_t, _ = self.lstm(src)
        return self.classifier(h_t)
