import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.labels = np.loadtxt(fname='dataset/pika/%s.csv' % mode,
                                 delimiter=',')

    def __getitem__(self, idx):
        x = Image.open('dataset/pika/%s/%s.jpg' %
                       (self.mode, idx)).convert('RGB')
        x = np.array(x)
        # [256, 256, 3] -> [3,256, 256]
        x = x.transpose((0, 2, 1))
        x = x.transpose((1, 0, 2))

        x = torch.tensor(x)
        x = x.float()

        y = torch.FloatTensor(self.labels[idx])

        return x, y

    def __len__(self):
        return len(self.labels)
