import torch
import torch.utils.data as data

from models.loss import get_truth, get_loss
from models.utils import show, get_anchor
from dataset import MyDataset
from models.full_model import Model


def train():
    net = Model()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练
    for epoch in range(20):
        net.train()
        for i, (x, y) in enumerate(loader_train):
            optimizer.zero_grad()
            # 预测
            # [5444, 4],[32, 5444, 2],[32, 21776]
            anchor, label_pred, offset_pred = net(x)

            # 获取每个anchor是否激活,偏移量,标签
            # [32, 5444],[32, 21776],[32, 21776]
            label, offset, masks = get_truth(anchor, y)

            # 计算loss
            loss = get_loss(label_pred, offset_pred, label, offset, masks)
            loss.mean().backward()
            optimizer.step()

            if i % 10 == 0:
                print(epoch, i, loss.mean().item())

    torch.save(net, './checkpoint/ssd.model')


if __name__ == '__main__':
    # 加载数据
    loader_train = data.DataLoader(dataset=MyDataset(mode='train'),
                                   batch_size=32,
                                   shuffle=True,
                                   drop_last=True)

    loader_test = data.DataLoader(dataset=MyDataset(mode='test'),
                                  batch_size=32,
                                  shuffle=True,
                                  drop_last=True)

    train()
