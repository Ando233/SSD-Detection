import torch
import torch.utils.data as data

from dataset import MyDataset
from models.utils import show


# 偏移量变换公式的逆运算
def inverse_offset(anchor, offset):
    # anchor -> [4]
    # offset -> [4]

    # x0,y0,x1,y0转换为cent_x,cent_y,w,h
    anchor_center = torch.empty(4)
    anchor_center[0] = (anchor[0] + anchor[2]) / 2
    anchor_center[1] = (anchor[1] + anchor[3]) / 2
    anchor_center[2] = anchor[2] - anchor[0]
    anchor_center[3] = anchor[3] - anchor[1]

    pred = torch.empty(4)

    # offset.x = (target.x - anchor.x) / anchor.w * 10
    # pred.x = offset.x * anchor.w * 0.1 + anchor.x
    pred[0] = (offset[0] * anchor_center[2] * 0.1) + anchor_center[0]

    # offset.y = (target.y - anchor.y) / anchor.h * 10
    # pred.y = offset.y * anchor.h * 0.1 + anchor.y
    pred[1] = (offset[1] * anchor_center[3] * 0.1) + anchor_center[1]

    # offset.w = log(tagret.w / anchor.w) * 5
    # pred.w  = exp(offset.w / 5) * anchor.w
    pred[2] = torch.exp(offset[2] / 5) * anchor_center[2]

    # offset.h = log(tagret.h / anchor.h) * 5
    # pred.h  = exp(offset.h / 5) * anchor.h
    pred[3] = torch.exp(offset[3] / 5) * anchor_center[3]

    # cent_x,cent_y,w,h转换为x0,y0,x1,y0
    pred_corner = torch.empty(4)
    pred_corner[0] = pred[0] - 0.5 * pred[2]
    pred_corner[1] = pred[1] - 0.5 * pred[3]
    pred_corner[2] = pred[0] + 0.5 * pred[2]
    pred_corner[3] = pred[1] + 0.5 * pred[3]

    return pred_corner


def predict(x, device):
    net.eval()

    # [3, 256, 256] -> [1, 3, 256, 256]
    x = x.unsqueeze(dim=0)

    # [5444, 4],[1, 5444, 2],[1, 21776]
    anchor, label_pred, offset_pred = net(x)

    # [1, 21776] -> [5444, 4]
    offset_pred = offset_pred.reshape(-1, 4)

    # 偏移量变换公式的逆运算
    # [5444, 4] -> [5444, 4]
    anchor_pred = torch.empty(5444, 4, device=device)
    for i in range(5444):
        anchor_pred[i] = inverse_offset(anchor[i], offset_pred[i])

    # softmax,让背景的概率和皮卡丘的概率相加为1
    # [1, 5444, 2] -> [1, 5444, 2]
    label_pred = torch.nn.functional.softmax(label_pred, dim=2)

    # 取anchor中物体是皮卡丘的概率
    # [1, 5444, 2] -> [5444]
    label_pred = label_pred[0, :, 1]

    # 只保留皮卡丘的概率高于阈值的结果
    anchor_pred = anchor_pred[label_pred > 0.1]
    label_pred = label_pred[label_pred > 0.1]

    return anchor_pred, label_pred


if __name__ == '__main__':
    net = torch.load('./checkpoint/ssd.pt')
    loader_test = data.DataLoader(dataset=MyDataset(mode='test'),
                                  batch_size=32,
                                  shuffle=True,
                                  drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, (x, y) in enumerate(loader_test):
        x = x.to(device)
        y = y.to(device)
        break

    for i in range(10):
        anchor_pred, label_pred = predict(x[i], device)
        anchor_pred = anchor_pred.to(device)
        label_pred = label_pred.to(device)
        if len(anchor_pred) == 0:
            print('not found')
            continue
        show(x[i], y[i], anchor_pred)
