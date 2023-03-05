import os

import cv2
import torch
from torch.utils.data import Dataset

from models.utils import transform_target


class MyDataset(Dataset):
    def __init__(self, mode, image_names, dataset_root):
        self.mode = mode
        self.image_names = image_names
        self.dataset_root = dataset_root

        if mode == 'train':
            self.image_path = os.path.join('%s' % self.dataset_root, 'train', 'images', '%s.jpg')
            self.anno_path = os.path.join('%s' % self.dataset_root, 'train', 'annotations', '%s.txt')
        elif mode == 'test':
            self.image_path = os.path.join('%s' % self.dataset_root, 'test', '%s.jpg')
            self.anno_path = os.path.join('%s' % self.dataset_root, 'test', 'annotations', '%s.txt')
        else:
            print('No phase!')

        # 按数据名字读取数据
        self.image_ids = list()
        with open(self.image_names, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.image_ids.append(line.strip('\n'))

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        img = cv2.imread(self.image_path % image_id)
        if img is None:
            print('No Image!')

        height, width, channels = img.shape
        target = transform_target(self.anno_path % image_id, width, height)

        img = cv2.resize(img, (300, 300))
        # [300, 300, 3] -> [3,300, 300]
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, target

    def __len__(self):
        return len(self.image_ids)
