import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def one_chann_normalize():
    # 1. 读取图像
    img = Image.open("ch2/assets/lena.png")
    # 转化为灰度图
    img = img.convert("L")


    # 2. 转化为张量（范围 [0,1]）
    to_tensor = transforms.ToTensor()   # [H,W,C] -> [C,H,W] 且值归一化到 [0,1]
    img_tensor = to_tensor(img)  # shape: [1,H,W]

    # 计算均值和标准差（来自该图像）
    mean = img_tensor.mean().item()
    std = img_tensor.std().item()

    # 3. 定义 Normalize（灰度只有一个通道）
    normalize = transforms.Normalize(mean=mean, std=std)  
    img_norm = normalize(img_tensor)  # 标准化： (x-0.5)/0.5 -> 映射到 [-1,1]

    # 4. 转回图像（把张量恢复到 [0,1] 再可视化）
    # 标准化后的值可能超出 [0,1]，所以先反归一化一下
    #denorm = transforms.Normalize(mean=[-1 * 0.5/0.5], std=[1/0.5])  # 反归一化
    #img_denorm = denorm(img_norm)

    #这里不反归一化，直接输出归一化以后的图像
    img_denorm = img_norm

    to_pil = transforms.ToPILImage()
    img_out = to_pil(img_denorm.clamp(0, 1))  # 约束范围到 [0,1]

    # 5. 显示标准化后的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(img_out, cmap="gray")
    plt.axis("off")
    plt.title("Normalized Gray Image (Restored for display)")
    plt.show()

    # 6. 保存标准化后的图像
    img_out.save("ch2/assets/lena_norm.png")
    print("保存完成：ch2/assets/lena_norm.png")

if __name__ == '__main__':
    one_chann_normalize()