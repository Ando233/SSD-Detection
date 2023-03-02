import torch
import torch.nn as nn
from models.model_first import FirstModel
from models.model_last import LastModel
from models.model_middle import MiddleModel


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = FirstModel()
        self.middle_1 = MiddleModel(c_in=64,
                                    anchor_size_small=0.37,
                                    anchor_size_big=0.447)
        self.middle_2 = MiddleModel(c_in=128,
                                    anchor_size_small=0.54,
                                    anchor_size_big=0.619)
        self.middle_3 = MiddleModel(c_in=128,
                                    anchor_size_small=0.71,
                                    anchor_size_big=0.79)
        self.last = LastModel()

    def forward(self, x):
        #定位框
        anchor = [None] * 5
        #类别
        label = [None] * 5
        #偏移量
        offset = [None] * 5

        #[2, 3, 256, 256] -> [2, 64, 32, 32],[4096, 4],[2, 8192],[2, 16384]
        x, anchor[0], label[0], offset[0] = self.first(x)

        #[2, 64, 32, 32] -> [2, 128, 16, 16],[1024, 4],[2, 2048],[2, 4096]
        x, anchor[1], label[1], offset[1] = self.middle_1(x)

        #[2, 128, 16, 16] -> [2, 128, 8, 8],[256, 4],[2, 512],[2, 1024]
        x, anchor[2], label[2], offset[2] = self.middle_2(x)

        #[2, 128, 8, 8] -> [2, 128, 4, 4],[64, 4],[2, 128],[2, 256]
        x, anchor[3], label[3], offset[3] = self.middle_3(x)

        #[2, 128, 4, 4] -> [2, 128, 1, 1],[4, 4],[2, 8],[2, 16]
        x, anchor[4], label[4], offset[4] = self.last(x)

        #[4096+1024+256+64+4, 4] -> [5444, 4]
        anchor = torch.cat(anchor, dim=0)

        #[2, 8192+2048+512+128+8] -> [2, 10888]
        label = torch.cat(label, dim=1)

        #[2, 10888] -> [2, 5444, 2]
        label = label.reshape(label.shape[0], -1, 2)

        #[2, 16384+4096+1024+256+16] -> [2, 21776]
        offset = torch.cat(offset, dim=1)
        return anchor, label, offset