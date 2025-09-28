# 残差网络 ResNet

## 1 问题引入

自首个基于 CNN 的架构（AlexNet）赢得 2012 年 ImageNet 竞赛以来，每个后续获胜的架构都在深度神经网络中使用更多层来降低错误率。这种方法在层数较少时有效，但当层数增加时，相同训练轮次时训练的效果下降反而下降了

![img](assets\abc.jpg)

这并不是“过拟合”，而是 **退化问题（Degradation Problem）** —— 更深的模型理论上应该至少能达到浅层模型的效果，但在实际训练中，性能反而下降。

目前 CV 领域的普遍认知是，高错误率是由梯度消失/爆炸引起的。由于每层之间参数变化幅度是以 `M*N` 的形式叠加的。因此，当增加层数时，更容易产生梯度消失或爆炸的问题





## 2 残差网络

ResNet 是由微软研究院的研究人员于 2015 年提出的，它引入了一种名为残差网络 (Residual Network) 的新架构。 

ResNet 提出的关键思想是：
 与其让每一层直接学习一个目标映射 $H(x)$，不如让它学习一个 **残差函数**：
$$
F(x) = H(x) - x \quad \Rightarrow \quad H(x) = F(x) + x
$$
这样，网络只需要学习输入和目标之间的 **差异**，而不是直接逼近目标映射。

- 这个 $H(x)$ 是“理想的映射”，也就是这一层应该学到的变换。例如在传统 CV 网络中，网络用卷积+BN+ReLU 的组合直接近似 $H(x)$



### 2.1 残差块

为了实现使网络学习残差而不是学习真实映射，论文中提出了 **残差块（Residual Block）** 的结构：

标准形式：
$$
y = F(x, \{W_i\}) + x
$$

- $x$：输入
- $F(x, \{W_i\})$：经过若干卷积、BN、ReLU 的变换
- $y$：输出

其中最核心的是 **shortcut connection（跳跃连接）**，即将输入 $x$ 直接加到输出 $y$ 上。

这样设计使得**残差函数 $F(x)$ 成为了学习的主体，而 shortcut connection 提供了“恒等基线”**，达成了预设目标

这种设计的好处有好处有：

- 如果某层的最佳功能就是“保持输入不变”（恒等映射），普通网络很难学到（要让卷积核接近单位矩阵）。但在残差单元里，只要学到 $F(x)=0$，就能轻松实现 $y=x$。
- 如果深层网络不好训练，残差块至少能让网络学到“恒等映射”（即 $F(x)=0, y=x$），避免性能退化。





### 2.2 使用例子

2015 年的原始论文提出了两种基本模块

**Plain Residual Block（适合浅层网络，如 ResNet-34）**

- 结构：两层 3×3 卷积 + BN + ReLU，再加上跳跃连接。

- 输出公式：
  $$
  y = x + W_2 \sigma(W_1 x)
  $$

**Bottleneck Residual Block（适合更深网络，如 ResNet-50/101/152）**

- 结构：1×1 降维卷积 → 3×3 卷积 → 1×1 升维卷积。
- 这样减少计算量，适合构建超深网络。



下面给出 ResNet-34 和34层卷积网络堆叠的对比图

可以看到**残差块不一定是单层结构，可以将多层网络的复核结构作为残差块的残差变换**

![灯箱](D:\data\repos\pytorch_learning\assets\ResNet.png)

原始论文的测试结果为：

- 在 **ImageNet 分类任务** 上：

  - ResNet-34 vs Plain-34：ResNet 明显更低的训练误差与测试误差。

  - ResNet-152 在 ImageNet 上取得了 **3.57% top-5 error**，当时是 SOTA。

- 在 **COCO 检测任务** 上，ResNet 作为 backbone 也取得了很大提升。

而 ResNet **最重要的贡献**是 ResNet 使得 100 层以上的网络第一次可以顺利训练并显著提升性能。



## 3 残差网络如何平缓梯度变化

网络由 $L$ 个残差块叠加，每个残差块的前向映射写成
$$
x_{k+1}=x_k+F_k(x_k),
\qquad k=0,\dots,L-1.
$$

- 这是残差网络的基本结构：每一层不是直接学习输出 $ x_{k+1} $，而是学习一个“残差” $ F_k(x_k) $，然后加到输入上。
- $ x_0 $ 是网络的初始输入，$ x_L $ 是最终输出。
- 每个 $ F_k(\cdot) $ 是第 $ k $ 个残差块中的非线性变换（比如卷积 + ReLU）。

设：

- 输入 $ x_k \in \mathbb{R}^n $（即 $ x_k $ 是一个 $ n $-维列向量）
- 输出 $ F_k(x_k) \in \mathbb{R}^m $（即残差函数输出是 $ m $-维列向量）

那么，每块的 Jacobian 矩阵 $ J_k $ 是一个 $ m \times n $ 的实数矩阵，定义为：

$$
J_k = \frac{\partial F_k(x_k)}{\partial x_k} =
\begin{bmatrix}
\frac{\partial F_{k,1}}{\partial x_{k,1}} & \frac{\partial F_{k,1}}{\partial x_{k,2}} & \cdots & \frac{\partial F_{k,1}}{\partial x_{k,n}} \\
\frac{\partial F_{k,2}}{\partial x_{k,1}} & \frac{\partial F_{k,2}}{\partial x_{k,2}} & \cdots & \frac{\partial F_{k,2}}{\partial x_{k,n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial F_{k,m}}{\partial x_{k,1}} & \frac{\partial F_{k,m}}{\partial x_{k,2}} & \cdots & \frac{\partial F_{k,m}}{\partial x_{k,n}}
\end{bmatrix}
\in \mathbb{R}^{m \times n}
$$
它描述了残差块 $ F_k $ 在输入点 $ x_k $ 处的局部线性变化率。其中 $F_{k,i}$ 指的是函数 $ F_k(x_k) $ 输出的第 $i$ 个分量， $x_{k,i}$ 同理。

我们想求的是：损失函数 $ \mathcal{L} $ 对最开始输入 $ x_0 $ 的梯度，即 $ \frac{\partial \mathcal{L}}{\partial x_0} $。根据链式法则，我们需要沿着整个网络反向传播：
$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_{L-1}} \cdot \frac{\partial x_{L-1}}{\partial x_{L-2}} \cdots \frac{\partial x_1}{\partial x_0}
$$

现在看每个 $ \frac{\partial x_{k+1}}{\partial x_k} $：

由前向公式 $ x_{k+1} = x_k + F_k(x_k) $，对 $ x_k $ 求导：

$$
\frac{\partial x_{k+1}}{\partial x_k} = I + \frac{\partial F_k(x_k)}{\partial x_k} = I + J_k
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot (I + J_{L-1}) \cdot (I + J_{L-2}) \cdots (I + J_0)
$$

> 注意乘法顺序！

可以注意到，每个因子都是 $ I + J_k $，而不是单纯的 $ J_k $ ，在传统深层网络中，如果没有残差连接，前向传播是：

$$
x_{k+1} = F_k(x_k) \Rightarrow \frac{\partial x_{k+1}}{\partial x_k} = J_k
$$

那么梯度就是：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot J_{L-1} J_{L-2} \cdots J_0
$$

如果这些 $ J_k $ 的奇异值都小于 1（常见情况），乘积会指数级衰减 ，导致梯度消失。但在残差网络中，每个因子是 $ I + J_k $，它包含了一个单位矩阵项，意味着即使 $ J_k $ 很小或接近零，这个因子也不会“坍缩”到零，至少保留了“恒等路径”的贡献。