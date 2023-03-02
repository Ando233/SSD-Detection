import torch.nn as nn
from models.utils import get_anchor


class FirstModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # label预测定位框中的内容
        self.label = nn.Conv2d(in_channels=64,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)

        # offset预测定位框的位置
        self.offset = nn.Conv2d(in_channels=64,
                                out_channels=16,
                                kernel_size=3,
                                padding=1)

    def forward(self, x):
        # [2, 3, 256, 256] -> [2, 64, 32, 32]
        x = self.cnn(x)

        # [2, 64, 32, 32] -> [4096, 4]
        anchor = get_anchor(image_size=32,
                            anchor_size_small=0.2,
                            anchor_size_big=0.272)

        # [2, 64, 32, 32] -> [2, 8, 32, 32]
        label = self.label(x)
        # [2, 8, 32, 32] -> [2, 32, 32, 8]
        label = label.permute(0, 2, 3, 1)
        # [2, 32, 32, 8] -> [2, 8192]
        label = label.flatten(start_dim=1)

        # [2, 64, 32, 32] -> [2, 16, 32, 32]
        offset = self.offset(x)
        # [2, 16, 32, 32] -> [2, 32, 32, 16]
        offset = offset.permute(0, 2, 3, 1)
        # [2, 32, 32, 16] -> [2, 16384]
        offset = offset.flatten(start_dim=1)

        return x, anchor, label, offset
