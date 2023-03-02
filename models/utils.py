import torch
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image
import PIL.ImageDraw


# 生成image_size**2个大框,image_size**2 *3个小框,均匀分布在图片上
def get_anchor(image_size, anchor_size_small, anchor_size_big):
    # 0-1等差数列,但不包括0和1,数量是image_size个
    # [0.2500, 0.7500]
    step = (np.arange(image_size) + 0.5) / image_size

    # 生成中心点,数量是image_size**2
    # [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
    point = []
    for i in range(image_size):
        for j in range(image_size):
            point.append([step[i], step[j]])

    # 根据中心点,生成所有的坐标
    anchors = []
    for i in range(len(point)):
        # 计算大正方形的4个坐标点,分别是中心点和宽高的一半做加减
        x0 = point[i][0] - anchor_size_big / 2
        y0 = point[i][1] - anchor_size_big / 2
        x1 = point[i][0] + anchor_size_big / 2
        y1 = point[i][1] + anchor_size_big / 2
        anchors.append([x0, y0, x1, y1])

        # 同上,计算小正方形的坐标点
        x0 = point[i][0] - anchor_size_small / 2
        y0 = point[i][1] - anchor_size_small / 2
        x1 = point[i][0] + anchor_size_small / 2
        y1 = point[i][1] + anchor_size_small / 2
        anchors.append([x0, y0, x1, y1])

        # 计算小长方形的坐标点
        x0 = point[i][0] - anchor_size_small * (2.0 ** 0.5) / 2
        y0 = point[i][1] - anchor_size_small / (2.0 ** 0.5) / 2
        x1 = point[i][0] + anchor_size_small * (2.0 ** 0.5) / 2
        y1 = point[i][1] + anchor_size_small / (2.0 ** 0.5) / 2
        anchors.append([x0, y0, x1, y1])

        # 计算另一个小长方形的坐标点
        x0 = point[i][0] - anchor_size_small * (0.5 ** 0.5) / 2
        y0 = point[i][1] - anchor_size_small / (0.5 ** 0.5) / 2
        x1 = point[i][0] + anchor_size_small * (0.5 ** 0.5) / 2
        y1 = point[i][1] + anchor_size_small / (0.5 ** 0.5) / 2
        anchors.append([x0, y0, x1, y1])

    anchors = torch.FloatTensor(anchors)

    return anchors


# 在图片上画出anchor
# x
def show(x, y, anchor):
    x = x.detach().numpy()
    x = x.astype(np.uint8)

    # (3, 256, 256) -> (256, 256, 3)
    x = x.transpose((1, 0, 2))
    x = x.transpose((0, 2, 1))

    y = y.detach().numpy()
    y = y * 256.0

    image = PIL.Image.fromarray(x)
    draw = PIL.ImageDraw.Draw(image)

    # 因为anchor的值域是0-1,需要转换到实际的图片尺寸.
    anchor = anchor.detach().numpy() * 256

    # 画框
    for i in range(len(anchor)):
        draw.rectangle(xy=anchor[i], outline='black', width=2)

    draw.rectangle(xy=y, outline='white', width=2)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    show(x[0], y[0], anchor)
