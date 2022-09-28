from torch.nn import functional as F


def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    x = F.silu(gate) * x
    return x
