# --------------------------------------------------------------------------------------------------------------------------
# Description: This script is about discriminator, taken from https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/model.py
# --------------------------------------------------------------------------------------------------------------------------
#

import torch.nn as nn

class Discriminator_bn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            # torch.nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            # torch.nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            # nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
