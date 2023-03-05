import argparse

import torch
import torch.utils.data as data

from config import HiXray
from models.loss import get_truth, get_loss
from models.utils import show, get_anchor
from dataset import MyDataset
from models.full_model import Model

HiXray_ROOT = "D:\Pytorch Project\SSD\dataset"
HiXray_NAMES = "D:\Pytorch Project\SSD\dataset"

# --dataset_root 目标数据集根路径
# --batch_size 训练时的batch_size
# --num_workers 数据加载时使用几个进程
# --cuda 指定cuda序号
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--mode', default='test', choices=['train', 'test'],
                    type=str, help='train or test')
parser.add_argument('--dataset_names', default=HiXray_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--dataset_root', default=HiXray_NAMES,
                    help='Dataset names txt path')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in data-loading')
parser.add_argument('--cuda', default=True, type=int,
                    help='Use CUDA to train model')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device(
            f"cuda:{args.cuda}"
        )
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')


def train():
    if args.dataset == 'HiXray':
        print('\nXray\n')
        cfg = HiXray
    net = Model().to(device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练
    for epoch in range(20):
        net.train()
        for i, (x, y) in enumerate(loader_train):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            # 预测
            # [5444, 4],[32, 5444,  2],[32, 21776]
            anchor, label_pred, offset_pred = net(x)
            anchor = anchor.to(device)
            label_pred = label_pred.to(device)
            offset_pred = offset_pred.to(device)

            # 获取每个anchor是否激活,偏移量,标签
            # [32, 5444],[32, 21776],[32, 21776]
            label, offset, masks = get_truth(anchor, y, device)
            label = label.to(device)
            offset = offset.to(device)
            masks = masks.to(device)

            # 计算loss
            loss = get_loss(label_pred, offset_pred, label, offset, masks, device)
            loss.mean().backward()
            optimizer.step()

            if i % 10 == 0:
                print(epoch, i, loss.mean().item())
                torch.save(net, './checkpoint/ssd.pth')


if __name__ == '__main__':
    # 加载数据
    loader_train = data.DataLoader(dataset=MyDataset(args.mode, args.dataset_names, args.dataset_root),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    train()
