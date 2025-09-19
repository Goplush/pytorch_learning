import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def one_chann_conv():

    #读取图像
    img = Image.open("ch2/assets/lena.png")
    #转化为灰度图
    img = img.convert("L")
    #转化为NP数组
    imgarr = np.array(img,dtype=np.float32)

    #先显示图片，显示后会阻塞，关闭后程序继续执行
    plt.figure(figsize=(6,6))
    #将colormap设置至灰度图
    plt.gray()
    plt.imshow(imgarr)
    plt.axis("off")
    plt.show()

    #获取图片的尺寸
    imh,imw=imgarr.shape
    #将数组转化为张量
    imgarr_t = torch.from_numpy(imgarr).reshape(1,1,imh,imw)
    #定义边缘卷积核，并将其维度处理为1*1*5*5
    kersize=5
    ker=torch.ones(kersize,kersize,dtype=torch.float32)*-1
    ker[2,2]=24  # 中心增强
    ker=ker.reshape(1,1,kersize,kersize)
    #设置卷积对象
    #输出通道设为2是为了留出一个由随机卷积核卷积的输出特征图来进行对比
    conv=torch.nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(kersize,kersize),bias=False)
    #设置卷积时使用的核
    conv.weight.data[0]=ker

    #对灰度图进行卷积操作
    img_conv_out=conv(imgarr_t)

    #对卷积后特征图进行压缩
    img_conv_out_im=img_conv_out.data.squeeze()
    print("卷积后尺寸: ",img_conv_out_im.shape)

    #可视化卷积后图像
    plt.figure(figsize=(6,6))
    plt.subplot(1,2,1)
    plt.gray()
    plt.axis("off")
    plt.imshow(img_conv_out_im[0])
    plt.subplot(1,2,2)
    plt.gray()
    plt.axis("off")
    plt.imshow(img_conv_out_im[1])
    plt.show()
    plt.imsave("ch2/assets/conv.png",img_conv_out_im[0].numpy())
    plt.imsave("ch2/assets/rand_conv.png",img_conv_out_im[1].numpy())



def rgb_depthwise_edge_conv():

    # 读取图像
    img = Image.open("ch2/assets/lena.png")
    # 保持 RGB 彩色图像（不转灰度）
    img = img.convert("RGB")
    
    # 转化为 NumPy 数组 (H, W, C)
    imgarr = np.array(img, dtype=np.float32)  # shape: (512, 512, 3)

    # 获取图像尺寸
    imh, imw, c = imgarr.shape  # 应该是 512x512x3

    # 将数组从 (H, W, C) 转为 (C, H, W)，并添加 batch 维度 -> (1, 3, 512, 512)
    img_t = torch.from_numpy(imgarr.transpose(2, 0, 1)).unsqueeze(0)  # shape: (1, 3, 512, 512)

    # 定义边缘检测卷积核 (5x5)，中心强正，周围负（类似拉普拉斯高斯或锐化核）
    kersize = 5
    ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
    ker[2, 2] = 24  # 中心权重较大，用于突出边缘

    # 扩展卷积核到三维：输出通道=3, 输入通道=3, 分组=3 → 深度卷积
    # 我们要构造一个形状为 (3, 1, 5, 5) 的权重，用于分组卷积
    weight = torch.stack([ker] * 3)  # shape: (3, 5, 5) → 每个通道用相同边缘核
    weight = weight.unsqueeze(1)     # shape: (3, 1, 5, 5)

    # 设置分组卷积（depthwise convolution），每输入通道单独卷积
    conv = nn.Conv2d(in_channels=3,
                     out_channels=3,
                     kernel_size=kersize,
                     groups=3,        # 关键：实现逐通道卷积（depthwise）
                     bias=False,
                     padding=kersize//2)  # 加 padding 保持尺寸不变

    # 手动设置卷积核权重
    with torch.no_grad():
        conv.weight.data.copy_(weight)

    # 执行卷积操作
    with torch.no_grad():
        output = conv(img_t)  # shape: (1, 3, 512, 512)

    # 去除 batch 维度，并转为 (3, 512, 512) -> 即 R, G, B 各自的边缘特征图
    edge_maps = output.squeeze().cpu()  # shape: (3, 512, 512)
    print("卷积后特征图尺寸:", edge_maps.shape)

    # 可视化：使用 2x2 网格布局展示原图和三个通道的边缘图
    plt.figure(figsize=(10, 10))

    # 第一行，第一列：原图
    plt.subplot(2, 2, 1)
    plt.imshow(imgarr.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    # 第一行，第二列：Red 通道边缘
    plt.subplot(2, 2, 2)
    plt.imshow(edge_maps[0], cmap='gray')
    plt.title("Edge - Red Channel")
    plt.axis("off")

    # 第二行，第一列：Green 通道边缘
    plt.subplot(2, 2, 3)
    plt.imshow(edge_maps[1], cmap='gray')
    plt.title("Edge - Green Channel")
    plt.axis("off")

    # 第二行，第二列：Blue 通道边缘
    plt.subplot(2, 2, 4)
    plt.imshow(edge_maps[2], cmap='gray')
    plt.title("Edge - Blue Channel")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 保存结果
    plt.imsave("ch2/assets/edge_r.png", edge_maps[0].numpy(), cmap='gray')
    plt.imsave("ch2/assets/edge_g.png", edge_maps[1].numpy(), cmap='gray')
    plt.imsave("ch2/assets/edge_b.png", edge_maps[2].numpy(), cmap='gray')

if __name__ == '__main__':
    #one_chann_conv()
    rgb_depthwise_edge_conv()