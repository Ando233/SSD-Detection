import torch.nn as nn

from models.utils import get_anchor


class LastModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.label = nn.Conv2d(in_channels=128,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)

        self.offset = nn.Conv2d(in_channels=128,
                                out_channels=16,
                                kernel_size=3,
                                padding=1)

    def forward(self, x):
        # [2, 128, 4, 4] -> [2, 128, 1, 1]
        x = self.cnn(x)

        # [2, 128, 1, 1] -> [4, 4]
        anchor = get_anchor(image_size=1,
                            anchor_size_small=0.88,
                            anchor_size_big=0.961)

        # [2, 128, 1, 1] -> [2, 8, 1, 1]
        label = self.label(x)
        # [2, 8, 1, 1] -> [2, 1, 1, 8]
        label = label.permute(0, 2, 3, 1)
        # [2, 1, 1, 8] -> [2, 8]
        label = label.flatten(start_dim=1)

        # [2, 128, 1, 1] -> [2, 16, 1, 1]
        offset = self.offset(x)
        # [2, 16, 1, 1] -> [2, 1, 1, 16]
        offset = offset.permute(0, 2, 3, 1)
        # [2, 1, 1, 16] -> [2, 16]
        offset = offset.flatten(start_dim=1)

        return x, anchor, label, offset
