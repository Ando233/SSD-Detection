import torch
import torch.nn as nn
from models.utils import get_anchor


class MiddleModel(nn.Module):
    def __init__(self, c_in, anchor_size_small, anchor_size_big):
        super().__init__()

        self.anchor_size_small = anchor_size_small
        self.anchor_size_big = anchor_size_big

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=c_in,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.label = nn.Conv2d(in_channels=128,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)

        self.offset = nn.Conv2d(in_channels=128,
                                out_channels=16,
                                kernel_size=3,
                                padding=1)

    def forward(self, x):
        # [2, 64, 32, 32] -> [2, 128, 16, 16]
        x = self.cnn(x)

        # [2, 128, 16, 16] -> [1024, 4]
        anchor = get_anchor(image_size=x.shape[-1],
                            anchor_size_small=self.anchor_size_small,
                            anchor_size_big=self.anchor_size_small)

        # [2, 128, 16, 16] -> [2, 8, 16, 16]
        label = self.label(x)
        # [2, 8, 16, 16] -> [2, 16, 16, 8]
        label = label.permute(0, 2, 3, 1)
        # [2, 16, 16, 8] -> [2, 2048]
        label = label.flatten(start_dim=1)

        # [2, 128, 16, 16] -> [2, 16, 16, 16]
        offset = self.offset(x)
        # [2, 16, 16, 16] -> [2, 16, 16, 16]
        offset = offset.permute(0, 2, 3, 1)
        # [2, 16, 16, 16] -> [2, 4096]
        offset = offset.flatten(start_dim=1)

        return x, anchor, label, offset
