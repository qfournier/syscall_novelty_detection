""" Inspired from Wangleiofficial:
https://github.com/pytorch/pytorch/issues/7455#issuecomment-720100742
Alternative implementation from NVIDIA:
https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527
fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing
.py#L18
"""

import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.epsilon = 0 if label_smoothing is None else label_smoothing

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction="mean")
        return self.epsilon * (loss / n) + (1 - self.epsilon) * nll
