import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
ker[2,2]=24
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