import torch
import torch.nn as nn


# 求n个anchor和target的交并比
def get_iou(anchor, target):
    # anchor -> [4, 4]
    # target -> [4]

    # x1-x0=宽
    # y1-y0=高
    # 宽*高=面积
    anchor_w = anchor[:, 2] - anchor[:, 0]
    anchor_h = anchor[:, 3] - anchor[:, 1]
    # [4]
    anchor_s = anchor_w * anchor_h

    # y部分的计算同理,形状都是[1],也就是标量
    target_w = target[2] - target[0]
    target_h = target[3] - target[1]
    target_s = target_w * target_h

    # 求重叠部分坐标
    cross = torch.empty(anchor.shape)

    # 左上角坐标取最大值,也就是取最右,下的点
    cross[:, 0] = torch.max(anchor[:, 0], target[0])
    cross[:, 1] = torch.max(anchor[:, 1], target[1])

    # 右下角坐标取最小值,也就是取最左,上的点
    cross[:, 2] = torch.min(anchor[:, 2], target[2])
    cross[:, 3] = torch.min(anchor[:, 3], target[3])

    # 右下坐标-左上坐标=重叠部分的宽度,高度
    # 如果两个矩形完全没有重叠,这里会出现负数
    # 这里不允许出现负数,所以最小值是0
    cross_w = (cross[:, 2] - cross[:, 0]).clamp(min=0)
    cross_h = (cross[:, 3] - cross[:, 1]).clamp(min=0)

    # 宽和高相乘,等于重叠部分面积,当然,宽和高中任意一个为0,则面积为0
    cross_s = cross_w * cross_h

    # 求并集面积
    union_s = anchor_s + target_s - cross_s

    # 交并比,等于交集面积/并集面积
    return cross_s / union_s


# 计算anchor和target的偏移量
def get_offset(anchor, target):
    # anchor -> [4]
    # target -> [4]

    # 求出每个框的宽高
    anchor_w = anchor[2] - anchor[0]
    anchor_h = anchor[3] - anchor[1]

    # 求出每个框的中心坐标
    anchor_cx = anchor[0] + anchor_w / 2
    anchor_cy = anchor[1] + anchor_h / 2

    # target的操作同理
    target_w = target[2] - target[0]
    target_h = target[3] - target[1]

    target_cx = target[0] + target_w / 2
    target_cy = target[1] + target_h / 2

    # 计算中心点的误差
    offset_cx = (target_cx - anchor_cx) / anchor_w * 10
    offset_cy = (target_cy - anchor_cy) / anchor_h * 10

    # 计算宽高的误差
    offset_w = torch.log(1e-6 + target_w / anchor_w) * 5
    offset_h = torch.log(1e-6 + target_h / anchor_h) * 5

    # [1],[1],[1],[1] -> [4]
    offset = torch.tensor([offset_cx, offset_cy, offset_w, offset_h])

    return offset


# 求每个anchor是激活还是非激活
def get_active(anchor, target):
    # anchor -> [16, 4]
    # target -> [4]

    # 不是0就是1,激活的是1,非激活的是0
    active = torch.zeros(len(anchor), dtype=torch.long)

    # 求每个anchor和target的交并比
    iou = get_iou(anchor, target)

    # 大于阈值的active,最大值active,其他都是非active
    active[iou >= 0.5] = 1
    active[torch.argmax(iou)] = 1

    return active == 1


# 根据active,转换成0和1,显然,active的是1,非active的是0
def get_mask(active):
    # [16, 4]
    mask = torch.zeros(len(active), 4)
    # 激活的行设置为1
    mask[active, :] = 1
    return mask


# 根据active,计算每个激活的anchor的类别
def get_label(active):
    # [16]
    label = torch.zeros(len(active), dtype=torch.long)
    # 因为在我这份数据集中对象只有一个类别,所以不需要从target中取类别
    label[active] = 1
    return label


# 计算激活的anchor和target的offset
def get_active_offset(active, anchor, target):
    # active -> [16]
    # anchor -> [16, 4]
    # target -> [4]

    # [16, 4]
    offset = torch.zeros(len(active), 4)
    for i in range(len(active)):
        if (active[i]):
            offset[i, :] = get_offset(anchor[i], target)

    return offset


def get_truth(anchor, target):
    # anchor -> [16, 4]
    # target -> [2, 4]

    labels = []
    offsets = []
    masks = []
    for i in range(len(target)):
        # 求每个anchor是激活还是非激活
        # [16]
        active = get_active(anchor, target[i])

        # 根据active,转换成0和1,显然,active的是1,非active的是0
        # [16, 4]
        mask = get_mask(active)
        masks.append(mask.reshape(-1))

        # 根据active,计算每个激活的anchor的类别
        # [16]
        label = get_label(active)
        labels.append(label)

        # 计算激活的anchor和target的offset
        # [16, 4]
        offset = get_active_offset(active, anchor, target[i])
        offsets.append(offset.reshape(-1))

    # [2, 64]
    labels = torch.stack(labels)
    # [2, 64]
    offsets = torch.stack(offsets)
    # [2, 64]
    masks = torch.stack(masks)

    return labels, offsets, masks


# 求loss
def get_loss(label_pred, offset_pred, label, offset, masks):
    # label_pred -> [32, 5444, 2]
    # offset_pred -> [32, 21776]

    # label -> [32, 5444]
    # offset -> [32, 21776]
    # masks -> [32, 21776]

    # [32, 5444, 2] -> [174208, 2]
    label_pred = label_pred.reshape(-1, 2)

    # [32, 5444] -> [174208]
    label = label.reshape(-1)

    # [174208]
    get_loss_cls = nn.CrossEntropyLoss(reduction='none')
    loss_cls = get_loss_cls(label_pred, label)
    # [174208] -> [32, 5444]
    loss_cls = loss_cls.reshape(32, -1)
    # [32, 5444] -> [32]
    loss_cls = loss_cls.mean(dim=1)

    # [32, 21776] * [32, 21776] -> [32, 21776]
    offset_pred *= masks
    # [32, 21776] * [32, 21776] -> [32, 21776]
    offset *= masks

    # [32, 21776]
    get_loss_box = nn.L1Loss(reduction='none')
    loss_box = get_loss_box(offset_pred, offset)
    # [32, 21776] -> [32]
    loss_box = loss_box.mean(dim=1)

    # [32] + [32] = [32]
    loss = loss_cls + loss_box
    return loss
