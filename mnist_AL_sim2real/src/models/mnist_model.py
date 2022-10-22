from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

import src.models.mc_dropout as mc_dropout

class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes, pen_emb_dim=128):
        super().__init__(num_classes, pen_emb_dim)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv1_drop = mc_dropout.MCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = mc_dropout.MCDropout2d()
        self.fc1 = nn.Linear(1024, self.pen_emb_dim)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(self.pen_emb_dim, num_classes)

    def mc_forward_impl(self, input: Tensor, return_pen_emb=False):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        pen_emb = F.relu(self.fc1_drop(self.fc1(input)))
        logits = self.fc2(pen_emb)
        logits = F.log_softmax(logits, dim=1)

        if return_pen_emb:
            return logits, pen_emb
        else:
            return logits, None
