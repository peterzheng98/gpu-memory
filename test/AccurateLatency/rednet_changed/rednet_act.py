import torch
import torch.nn as nn


class RedNet(nn.Module):
    def __init__(self):
        super(RedNet, self).__init__()

    def forward(self, x):
        return (x > 0) * torch.log(x + 1)
