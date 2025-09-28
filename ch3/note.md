[TOC]



## 1 随机梯度下降

### 1.1 随机梯度下降与批次梯度下降

在机器学习中，我们通常希望找到一组模型参数（如神经网络的权重），使得模型在训练数据上的预测误差（即损失函数）最小。梯度下降法通过沿着损失函数梯度的反方向更新参数，逐步逼近最优解。

标准的**梯度下降（Gradient Descent）** 在每次更新参数时使用**整个训练集**计算梯度，虽然收敛稳定，但计算开销大，尤其在数据量庞大时效率较低。

SGD 对梯度下降进行了简化：**每次只使用一个训练样本（或一个小批量样本）来估计梯度并更新参数**。因此，它具有以下特点：

- **计算效率高**：每次更新只需计算一个样本的梯度，速度更快。
- **内存占用低**：不需要一次性加载全部数据。
- **具有随机性**：由于每次梯度是基于单个样本的噪声估计，更新路径会“抖动”，但这种噪声有时有助于跳出局部极小值。
- **收敛速度较快（初期）**，但可能在最优解附近震荡，不一定精确收敛。

以单个样本更新可学习参数为例，SGD 算法的更新规则为：

假设损失函数为 $ L(\theta; x^{(i)}, y^{(i)}) $，其中 $ \theta $ 是**代表模型所有可学习参数的向量**，$ (x^{(i)}, y^{(i)}) $ 是第 $ i $ 个训练样本。

SGD 的更新规则为：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta; x^{(i)}, y^{(i)})
$$

其中：

- $ \eta $ 是学习率（learning rate），控制更新步长；
- $ \nabla_\theta L $ 是损失函数对参数的梯度；
- 每次迭代随机选取一个样本 $ i $（或按顺序遍历）。

实际应用中，通常使用 **batch SGD**：每次使用一小批（如 32、64、128 个）样本来计算梯度。它结合了批量梯度下降的稳定性和 SGD 的效率，是深度学习中最常用的优化方式。

batch SGD的优点有：

- 计算开销小，适合大规模数据；
- 可在线学习（数据流式输入）；
- 随机性有助于逃离鞍点和局部极小值。



### 1.2 动量

**动量（Momentum）** 是对标准随机梯度下降（SGD）的一种重要改进，它能显著提升优化过程的**稳定性**和**收敛速度**。

SGD 对参数的优化过程就好像是一个球从山坡上滚下来：

**标准 SGD** 就像每次只根据当前位置的坡度（梯度）决定下一步怎么走——走一步、停一下、再看坡度、再走一步。路径会非常“抖动”，尤其在峡谷或崎岖地形中来回震荡，甚至有可能停留在局部极小值（比如马鞍面的鞍点）

为了解决这个问题，我们可以参考物理学中的动量，$p=m\times v$（质量 × 速度）。这里：

- $ v $ 类似速度；
- $\beta$  类似“保留多少原有动量”（ $\beta$ 越大，惯性越强）；
- 梯度相当于“外力”，改变运动方向。

所以，**动量让优化过程有了“记忆”和“惯性”**。

而**带动量的 SGD** 则像一个**有惯性的球**：它不仅看当前坡度，还保留了之前运动的方向和速度。即使当前坡度变小或方向突变，它仍会“惯性前冲”，从而：

- 减少震荡；
- 加速穿越平坦区域；
- 更快地冲向谷底。

在带动量的 SGD 中，梯度用来更新速度（动量），而用动量来更新可学习参数：
$$
v \leftarrow \beta v - \eta \nabla_\theta L\\
\theta \leftarrow \theta + v
$$

- $ v $：类似“速度”或“累积梯度”，是一个和 $ \theta $ 同维度的向量，初始通常设为 0；
- $ \beta $：动量系数（通常取 0.9 或 0.99），表示保留多少历史速度；
- $ \eta $：学习率。

如下图所示，动量的引入能达成如下效果

- 抵消小幅度震荡，快速通过“抖动”区域
- 历史速度累积，使得在梯度平缓区域也能有较高的学习效率
- 能有助于避免停留在局部极小值点

<img src="assets\5.png" alt="img" style="zoom: 33%;" />



### 1.3 Nesterov 动量

**Nesterov 动量**（Nesterov Accelerated Gradient，简称 NAG）是对标准动量（Momentum）方法的一种巧妙改进，由数学家 Yurii Nesterov 在 1983 年提出。它在理论上具有更优的收敛速度，并在实践中也常表现出更好的性能。

标准动量（Momentum）的更新逻辑是：

1. 先计算当前位置 $ \theta_t $ 的梯度；
2. 然后结合历史速度更新参数。

而 Nesterov 动量的关键改进在于：先根据历史动量**”预估“下一步的位置，在这个预估位置上计算梯度更新动量**。这样可以**提前感知前方地形变化**，避免冲过头（比如冲过谷底又反弹回来），从而减少震荡、加速收敛。

> 注意 NAG 只是提前感知地形，参数更新的起点还是初始位置

NAG 的方法如下

先做一个“试探性”更新（look-ahead）：
$$
\tilde{\theta}_t = \theta_t - \eta \beta v_{t-1}
$$
在试探位置 $ \tilde{\theta}_t $ 上计算梯度更新动量：

$$
v_t = \beta v_{t-1} + \nabla_\theta L(\tilde{\theta}_t)
$$
最后在**初始位置正式更新参数**：

$$
\theta_{t+1} = \theta_t - \eta v_t
$$
通过提前感知，NAG 可以通过未来的函数梯度更新现有的动量，原始论文已经证明了这在梯度平滑、噪音少的条件下效果很好。而且大量实验也证明了：带动量的方法（包括 NAG）**收敛更快、训练更稳定**；并且倾向于找到**更平坦的极小值（flat minima）**，这类解通常**泛化能力更强**（Hochreiter & Schmidhuber, 1997）。

但是在噪声很大的环境下，NAG会在更新决策中引入更多的噪声，从而对学习产生干扰



## 2 优化器

本部分笔者想以一个 Reddit 帖子的回复来开启本部分的笔记

> 目前人工智能，尤其是深度学习学科的现实是，我们仍然不太清楚神经网络是如何学习和泛化的，或者梯度下降(GD)/SGD何时有效、何时无效。有一些分析工具表明，过度参数化的网络可以找到几乎任何函数的不错近似（通常好的入手之作是 [《神经切线核》论文](https://arxiv.org/abs/1806.07572)；普遍逼近器的观点来自 [Hornik及其同事在80年代末的定理](https://www.sciencedirect.com/science/article/abs/pii/0893608089900208)），然而众所周知，为了让GD/SGD有效，我们需要“好的”架构（前向跳跃连接就是一个很好的例子）以及一些正则化——通常是dropout和使用较小的批量。 [《神经网络损失景观可视化》](https://proceedings.neurips.cc/paper/2018/file/a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf) 是一个关于这个主题的很棒的论文，其中有很多可视化，使这些想法更清晰。总体而言，我们现在认为“好的”架构和训练方案允许“最小值相连”的特性——也就是过度参数化允许从一个最小值平滑过渡到另一个最小值，这意味着几乎没有局部“陷阱最小值”。一篇关于这个主题的近期好论文是 [《深度学习中的连通子水平集》](https://arxiv.org/abs/1901.07417)。这意味着我们预计GD/SGD在良好定义的网络和训练方案中在寻找局部最小值方面会表现得相当好。
>
> 现在，你可能注意到上述论文的共同点是模型需要足够大才能很好地工作。在真实世界中，这意味着模型具有足够的表示能力（通常通过Vapnik–Chervonenkis维度来衡量）来记忆输入的样本。Bengio兄弟在我们对记忆的理解上已经进行了很长时间的来回讨论（一个好的历史参考是 [在这里](https://arxiv.org/pdf/1706.05394.pdf)，一个较新的参考是 [在这里](https://openreview.net/pdf?id=B1l6y0VFPr)），但我们开始意识到这个问题比我们想的要复杂得多（例如 [GPT-2在训练数据集中看过一次或两次后记住800多个数字的π](https://arxiv.org/pdf/2012.07805.pdf)）
>
> 因此，关于为什么SGD优于GD的最佳假设是两方面的。首先，通过引入噪声来防止完美记忆的发生，并迫使学习通过鲁棒特征来进行——这也是为什么小批量通常比大批量效果更好的原因。其次，它迫使网络学习数据的冗余误差纠正编码，使其检测能力更为鲁棒。
>
> **也就是说，SGD、泛化、良好的模型架构和良好的训练方案仍然是一个研究热点，目前还没有任何（可靠且完备的）结论可供参考。**



在训练开始的时候，参数会与最终的最优值点距离较远，所以需要使用较大的学习率；经过充分训练之后，为了避免参数向量在最优质附近震荡则需要更小的训练学习率。因此，在众多的优化算法中，不仅有通过改变更新时梯度方向和大小的算法，还有一些算法则是优化了学习率等参数的变化，如一系列自适应学习率的算法 Adadelta、RMSProp 及 Adam 等。

在 PyTorch 中的 `optim` 模块，提供了多种可直接使用的深度学习优化算法，内置算法包括 Adam、SGD、RMSprop 等，无须人工实现随机梯度下降算法，直接调用即可。可直接调用的优化算法类如表 3-1 所示。

表 3-1  PyTorch 中的优化器

| 类                     | 算法名称             |
| ---------------------- | -------------------- |
| `torch.optim.Adadelta` | Adadelta 算法        |
| `torch.optim.Adagrad`  | Adagrad 算法         |
| `torch.optim.Adam`     | Adam 算法            |
| `torch.optim.Adamax`   | Adamax 算法          |
| `torch.optim.ASGD`     | 平均随机梯度下降算法 |
| `torch.optim.LBFGS`    | L-BFGS 算法          |
| `torch.optim.RMSprop`  | RMSprop 算法         |
| `torch.optim.Rprop`    | 弹性反向传播算法     |
| `torch.optim.SGD`      | 随机梯度下降算法     |

它们又可以分为以下几类：

**基础梯度下降类（First-order Methods）**

这类算法直接使用梯度信息进行参数更新，是最基础的优化方法。

- **随机梯度下降（SGD）**：`torch.optim.SGD()`
- **平均随机梯度下降（ASGD）**：`torch.optim.ASGD()`
  （ASGD 是 SGD 的变体，通过平均历史参数来提升收敛稳定性）



**自适应学习率方法（Adaptive Learning Rate Methods）**

这类算法根据参数的历史梯度自动调整每个参数的学习率，适合处理稀疏数据或非平稳目标。

- **Adagrad**：`torch.optim.Adagrad()`
  （为频繁更新的参数分配较小学习率，稀疏参数分配较大学习率）
- **Adadelta**：`torch.optim.Adadelta()`
  （改进 Adagrad 的学习率衰减问题，使用滑动窗口累积梯度）
- **RMSprop**：`torch.optim.RMSprop()`
  （类似 Adadelta，使用指数加权移动平均）
- **Adam**：`torch.optim.Adam()`
  （结合动量（momentum）和 RMSprop 的思想，使用一阶矩和二阶矩估计）
- **Adamax**：`torch.optim.Adamax()`
  （Adam 的变体，使用无穷范数代替 L2 范数，对学习率更稳定）

特点：每个参数有独立的自适应学习率，通常收敛更快、调参更简单。



**启发式/非梯度比例更新方法**

这类算法不直接使用梯度的大小，而是仅利用梯度的符号或局部信息调整步长。

- **Rprop（弹性反向传播）**：`torch.optim.Rprop()`
  （仅使用梯度符号，根据梯度方向是否一致动态调整步长，适用于全批量场景）

 特点：忽略梯度幅值，只关注方向变化，适合小批量或全批量训练。 



 **拟牛顿法（Quasi-Newton Methods）**

这类算法近似二阶导数（Hessian 矩阵）信息，以加速收敛，**通常用于小规模问题**。

- **L-BFGS**：`torch.optim.LBFGS()`
  （Limited-memory BFGS，内存受限的拟牛顿法，适合确定性优化）

特点：利用曲率信息，收敛快但内存和计算开销大，通常用于小模型或微调。 



pytorch 框架为这些优化器提供了统一的使用接口和参数访问接口，会在本部分的最后进行介绍

### 2.1 Adagrad 与 Adadelta

#### 2.1.1 参数更新方法

**Adagrad (2011)** 是第一个被广泛应用于深度学习、并且系统性地提出了 **“为每个参数分配独立的自适应学习率”** 的优化算法。它的核心思想是：梯度大的参数学习率下降得更快，梯度小的参数下降得更慢，从而实现动态调整。而 Adadelta 则是 Adagrad 的优化版本。

Adagrad 的核心思想是：**对频繁更新的参数使用较小的学习率，对不常更新的参数使用较大的学习率**。

它维护一个历史梯度平方的累积和：

$$
G_t = \sum_{i=1}^{t} g_i^2
$$

然后更新参数：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t
$$

其中 $ \epsilon $ 是一个很小的数（如 $ 10^{-8} $），防止除零。

这个梯度的累计平方和是为每个可学习参数都进行独立维护的，所以上文说为每个参数分配独立的自适应学习率

但是 Adagrad 的问题是 $G_t$ 只增不减，导致学习率会越来越小，最终可能提前停止学习（“过早收敛”问题）。

为了解决这一问题，Adadelta **使用动态一望代替全历史累积**来记录梯度的历史变化，它不再累积从开始到现在的所有梯度平方，而是只保留最近若干步的信息，通过指数移动平均（EMA）实现。

具体而言，Adadelta 定义了梯度平方的指数移动平均：

$$
E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot g_t^2
$$

其中：

- $E[g^2]_t$ 为表示带权历史梯度平方和的符号
- $ \rho $ 是衰减率（通常取 0.9），类似动量中的参数。

这样，旧的梯度信息会逐渐被“遗忘”，避免了学习率无限衰减。

此外，AdaDelta 还观察到：Adagrad 的更新形式是

$$
\Delta\theta_t = -\frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

其中 $ \eta $ 仍然是一个需要手动调的超参数。AdaDelta 想：能不能用过去参数更新量的大小来自动决定当前步长？

于是，它也维护一个参数更新量平方的指数移动平均：

$$
E[\Delta\theta^2]_t = \rho \cdot E[\Delta\theta^2]_{t-1} + (1 - \rho) \cdot (\Delta\theta_{t-1})^2
$$

然后，AdaDelta 的更新规则为：

$$
\Delta\theta_t = -\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

$$
\theta_{t+1} = \theta_t + \Delta\theta_t
$$

- **分母** $ \sqrt{E[g^2]_t} $：衡量当前梯度的“典型大小” → 如果梯度一直很大，就走小步。
- **分子** $ \sqrt{E[\Delta\theta^2]_{t-1}} $：衡量过去参数更新

这样就用表示梯度变化的方式表示了学习率的变化，并**让模型以这种方式自动维护学习率，避免引入显式的学习率 η**，步长完全由历史梯度和历史更新量决定，自动平衡更新幅度，保持单位一致性（因为分子分母单位相同，结果无量纲）。

目前 **Adam 更常用**，但在某些场景（如 RNN 训练、学习率调参困难时），AdaDelta 仍是一个稳健的选择。



#### 2.1.2 pytorch 类

对于 `torch.optim.Adadelta` 类，其构造函数签名为：

```python
torch.optim.Adadelta(
    params,
    lr=1.0,
    rho=0.9,
    eps=1e-6,
    weight_decay=0
)
```

下面介绍一下参数含义：

- **`params`**：待优化的参数。可以是：一个 `iterable`（如列表），包含 `torch.Tensor` 类型的参数。但是**最常见的做法是直接传入 `model.parameters()`**。
- **`lr`**（learning rate）：类型：`float`，默认值：`1.0`。**全局学习率缩放因子**。虽然 AdaDelta 理论上自适应步长、无需手动设置学习率，但 PyTorch 保留此参数作为对更新量的统一缩放。通常保持默认值 `1.0` 即可，不建议大幅调整。
- **`rho`**：类型：`float`，默认值：`0.9`。**指数移动平均的衰减系数**，用于计算梯度平方和参数更新量平方的滑动平均（对应原始论文中的  $ \rho $ ）。值越接近 1，历史信息保留越久；常用值为 `0.9` 或 `0.95`。
- **eps**:类型：`float`，默认值：`1e-6`。**数值稳定项，加在分母中以防止除零错误**（即 $ \sqrt{E[g^2]_t + \epsilon} $ 中的 $ \epsilon $）。PyTorch 默认为 `1e-6`，而有些实现（如 TensorFlow）使用 `1e-8`；若训练不稳定，可尝试减小此值（如设为 `1e-8`）。
- **weight_decay**：类型：`float`，默认值：`0`。**正则化系数（权重衰减）**。若设为正数（如 `1e-4`），会在每次更新前对参数施加衰减：$ \theta \leftarrow \theta - \text{weight\_decay} \cdot \theta $。用于防止过拟合。



Adadelta 的优化方法内部变量会存储在 `optimizer.state` 中（字典结构），每个参数张量对应一个状态子字典

可以通过以下方式查看：

```
for group in optimizer.param_groups:
    for p in group['params']:
        print(optimizer.state[p].keys())
```

- `group`：
  - 在 PyTorch 中，每个优化器（如 `torch.optim.SGD`, `torch.optim.Adam`）都有一个属性 `param_groups`。`param_groups` 是一个 **list**，里面的每个元素是一个 **dict**，叫做 **参数组（parameter group）**。
  - 每个参数组都包含：`'params'`：要被优化的一组参数（通常是张量 `nn.Parameter` 的引用）。以及该组的优化超参数，比如 `'lr'`（学习率）、`'weight_decay'`、`'momentum'` 等。
- `p`：`group['params']` 是一个 **参数列表**，其中的每个元素 `p` 就是一个 **待优化参数张量**（`torch.nn.Parameter` 类型）。例如，一个 `nn.Linear` 层就有两个 `Parameter`：权重 `weight` 和偏置 `bias`，它们都会出现在 `param_groups` 里。
- `optimizer.state[p]`：`optimizer.state` 是一个字典，**键（key）**是参数张量 `p`，**值（value）**是该参数对应的优化器状态。
- 总而言之，上面的方法做的就是**对优化器负责的每一个用独立超参数组控制的参数组`param_groups` 的每一个优化器状态键的内容进行输出**

对于 `Adadelta`，每个参数 `p` 的状态通常包含以下两个缓冲区（buffers）：

-  `square_avg`
  - **含义**：即 $E[g^2]_t$ ，**梯度平方的指数移动平均**。
  - **形状**：与参数 `p` 相同。
- `acc_delta`
  - **含义**：即 $ E[\Delta\theta^2]_{t-1} $ ，**参数更新量平方的指数移动平均**。
  - **形状**：与参数 `p` 相同。
  - **用途**：用于计算当前步的自适应步长（分子部分）。

虽然可以通过 `optimizer.state` 查看 `square_avg` 和 `acc_delta`，但一般只用于调试或可视化，**训练中不应手动修改**。



### 2.2 RMSprop、Adam 和 Adamax 

#### 2.2.1 RMSprop 参数更新方法

RMSprop 由 Geoffrey Hinton 在 2012 年提出，旨在解决 AdaGrad 在训练后期学习率过小的问题。它通过使用指数加权移动平均（EWMA）来累积梯度的平方，从而对学习率进行自适应调整。它“遗忘”较早的梯度信息，避免学习率过早衰减。

它的更新公式为：

设参数为 $ \theta $，损失函数对参数的梯度为 $ g_t = \nabla_\theta \mathcal{L}(\theta_t) $。

- 梯度平方的移动平均：

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t^2
$$

其中 $ \beta \in [0, 1) $（通常取 0.9），$ v_0 = 0 $。

- 参数更新：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t
$$

其中 $ \eta $ 是初始学习率，$ \epsilon $ 是一个很小的常数（如 $ 10^{-8} $），用于防止除零。



#### 2.2.2 Adam 和 Adamax 参数更新方法

而 Adam 优化算法**融合了动量（Momentum）和 RMSprop 的优点**，其核心思想是同时维护梯度的一阶矩（均值）和二阶矩（未中心化的方差）的指数移动平均，并对它们进行偏差修正。

更新公式如下：

**一阶矩（动量项）**：  
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$



 **二阶矩（梯度平方）**：  
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$
这个二阶矩的维护方法，实际上参考了 Adadelta 的 EMA 方法，因此让Adam拥有了自适应调整学习率的能力



 **偏差修正（因为初始值为 0，早期估计有偏）**：  
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$


**参数更新**：  
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

>常用超参数：$ \beta_1 = 0.9 $，$ \beta_2 = 0.999 $，$ \epsilon = 10^{-8} $。

**优点**：

- 结合了动量和自适应学习率的优点。
- 通常收敛快、稳定，是目前最广泛使用的优化器之一。



Adam优化算法维护**损失函数梯度（即随机梯度估计）的一阶矩（均值）和二阶矩（未中心化的方差）的指数移动平均估计**的目的为：

| 组件                                                         | 功能概述                                                 | 对收敛的帮助                                                 |
| ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| **一阶矩估计（动量）**  $ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ | 平滑梯度方向，抑制震荡，加速穿越平坦区域                 | 减少训练过程中的“抖动”，使优化路径更平滑，尤其在损失曲面有“峡谷”或“锯齿”时效果显著 |
| **二阶矩估计（自适应学习率）** $  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ | 为每个参数单独调整学习率：梯度大则步长小，梯度小则步长大 | 自动平衡不同参数的更新速度，避免某些参数更新过快或过慢       |

- 动量平滑梯度方向；
- 二阶矩抑制大梯度带来的突变；



而 Adam 引入偏差修正的原因是因为 Adam 在初始化时经常取 $ m_0 = 0, v_0 = 0 $，导致早期估计严重偏向零（尤其在 $ \beta_1, \beta_2 $ 接近 1 时）。  为解决此问题，Adam 引入偏差修正：
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

在训练初期（$ t $ 较小），修正项显著放大 $ m_t $ 和 $ v_t $，避免学习率被过度压缩。随着 $ t $ 增大，修正项趋近于 1，影响消失。这使得 Adam 在训练早期也能保持合理的更新幅度，避免像 RMSprop 或原始动量那样在初期“启动慢”。



尽管 Adam 收敛快，但也有一些局限：

- **泛化能力有时不如 SGD**：在某些任务（如图像分类）中，SGD + 学习率衰减可能达到更低的测试误差。
- **可能收敛到尖锐极小值**：有研究认为 Adam 倾向于收敛到泛化性较差的“sharp minima”。
- **后期可能震荡**：因自适应学习率未显式衰减，有时需配合学习率调度（如 cosine decay）。



最后，Adamax 算法是 Adam 的简单变体，二者在同一篇论文中被提出，Adam 使用 $ L^2 $ 范数（平方）来估计梯度的幅度，而 Adamax 使用 $ L^\infty $ 范数（即梯度绝对值的最大值），这在某些情况下更稳定，尤其当梯度分布稀疏时。

Adamax的更新函数为：

**一阶矩（同 Adam）：**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$



**二阶矩（使用无穷范数）：**
$$
u_t = \max(\beta_2 u_{t-1}, |g_t|)
$$

> 注意这里不是平方，而是取绝对值的最大值。



**偏差修正（仅对一阶矩）：**
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$



**参数更新：**
$$
\theta_{t+1} = \theta_t - \frac{\eta}{u_t} \hat{m}_t
$$



优点：

- 对异常梯度更鲁棒。
- 更新更稳定，尤其在高维或稀疏梯度场景下。
  - “梯度稀疏”（Sparse Gradients）是深度学习和优化中的一个重要概念，指的是在某次参数更新中，**只有少数参数的梯度是非零的，而大多数参数的梯度为零（或接近零）**。换句话说，梯度向量中大部分元素为零，只有少量位置有显著的梯度值。

- 无需对 $ u_t $ 做偏差修正（因其单调不减）。

**实际使用建议**：

- **Adam** 是默认首选，适用于大多数任务。
- 如果训练不稳定或梯度稀疏，可尝试 **Adamax**。
- **RMSprop** 在循环神经网络（RNN）中仍有良好表现，尤其在早期深度学习实践中常用。



#### 2.2.3 pytorch类（以 Adam 为例）

最后，以Adam为例介绍一下pytorch的实现， `torch.optim.Adam` 构造函数签名为

```python
torch.optim.Adam(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False
)
```

- **`betas`**为一个包含两个浮点数的元组 `(beta1, beta2)`，分别用于控制一阶矩（梯度均值）和二阶矩（梯度平方均值）的指数移动平均衰减率。默认为 `(0.9, 0.999)`，对应论文中的推荐值。
- **`amsgrad`**决定了是否使用 **AMSGrad** 变体（ICLR 2018 提出）。若为 `True`，则在更新时使用历史二阶矩的最大值（而非当前值），以解决 Adam 可能不收敛的问题。默认为 `False`。



而在 Adam 的优化器状态同样存放在 `optimizer.state` 中，其中包括

- `step`：该参数已更新的步数（int）
- `exp_avg`：一阶矩估计（即 $ m_t $，形状同参数）
- `exp_avg_sq`：二阶矩估计（即 $ v_t $，形状同参数）
- （若 `amsgrad=True`，还会包含 `'max_exp_avg_sq'`）



#### 2.2.4 使用建议

Adam 对学习率相对不敏感，但 **`lr` 仍需合理设置**（常用范围：`1e-5` 到 `1e-3`）。

若模型过拟合，可尝试添加 **`weight_decay`**（如 `1e-4` 或 `1e-5`）。

在某些任务（如 Transformer 训练）中，常配合 **学习率预热（warmup）+ 衰减策略** 使用 Adam。

若怀疑 Adam 收敛不稳定，可尝试开启 **`amsgrad=True`**。



### 2.3 SGD 与 ASGD

#### 2.3.1 ASGD 方法

关于随机梯度下降（SGD），已经在第一部分有了详尽的介绍。虽然 SGD 计算高效、适合大规模数据，但它在收敛后期会在最优解附近震荡，尤其当学习率衰减不够快时，难以达到高精度解。

ASGD 通过**对迭代过程中的参数进行平均**，来平滑这些震荡，从而获得更稳定、更接近真实最优解的估计。

ASGD 在参数更新方式上与 SGD 相同，但是ASGD不直接使用最后一步的参数 $ \theta_T $，而是使用从某个时刻开始（或从头开始）所有参数的平均值作为最终模型参数。

最常见的是使用尾部平均（即从某个时间点 $ t_0 $ 开始平均，以减少初始不稳定阶段的影响）：

$$
\bar{\theta}_T = \frac{1}{T - t_0 + 1} \sum_{t=t_0}^{T} \theta_t
$$




#### 2.3.2 使用情况

**SGD（尤其是带动量的 SGD，如 SGD with Momentum）及其变体（如 Adam、RMSProp）的使用远比 ASGD 更广泛**。ASGD 虽然在理论上具有优势，但在实际应用中相对较少被直接使用，原因如下：

- **深度学习主导非凸优化**：现代主流任务（如图像分类、NLP、大模型训练）大多涉及**高度非凸的损失函数**。ASGD 的理论优势（如 Polyak–Ruppert 平均的渐近最优性）主要建立在**凸优化**假设之上，在非凸场景下缺乏强理论保证。
- **自适应优化器更受欢迎**：Adam、AdamW 等自适应学习率优化器因其**训练稳定、对超参不敏感、收敛快**，已成为深度学习默认选择。SGD with Momentum 也在某些领域（如计算机视觉）因泛化性能好而被保留。
- **ASGD 实现和调参较复杂**：ASGD 引入了额外超参数（如 `lambd`、`alpha`、`t0`），调优成本高，且默认值未必适用于所有任务。相比之下，SGD/Adam 的超参更直观（主要是 `lr` 和 `weight_decay`）。



尽管使用较少，ASGD 仍在**凸优化问题**中占据着牢固的生态位：如大规模线性模型（逻辑回归、SVM）、在线学习系统，ASGD 能提供稳定、高效的解。



#### 2.3.3 pytorch 类

以 `torch.optim.SGD` 类为例介绍一下 SGD 及其变体的pytorch实现

构造函数签名为：

```python
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

介绍一些参数：

- **`momentum`** ：动量因子（默认为`0`）。引入动量可加速 SGD 在相关方向上的更新，并抑制震荡。最好只取大于等于0的值

- **`dampening`**：阻尼因子（默认：`0`）。用于抑制动量项的累积，仅在 `momentum > 0` 时生效。梯度对动量的贡献会乘以 `(1 - dampening)`。通常设为 `0`（即无阻尼）

  - **一般不建议随意调整 `dampening`**，除非有特定理论或实验依据。
  - 具体而言，原先学习率兼任着着阻尼的作用，添加阻尼因子后，学习率作为更新参数时动量  $ v_t $ 的乘子

  $$
  v_t = \mu \cdot v_{t-1} + (1 - \delta) \cdot g_t\\
  \theta_t = \theta_{t-1} - \text{lr} \cdot v_t
  $$
  
  > 请注意将上面的公式与标准的带动量的SGD比较
  
- **`weight_decay`** ：权重衰减（L2 正则化）系数（默认：`0`）。在每次参数更新前，先对参数施加 L2 惩罚

  $\theta \leftarrow \theta - \text{lr} \cdot (\text{weight\_decay} \cdot \theta)$

- **`nesterov`**：决定是否使用 Nesterov 动量（默认：`False`）。若为 `True`，则使用 Nesterov accelerated gradient（NAG）形式的动量更新，通常能提供更好的收敛性。**仅在 `momentum > 0` 时有效**



而 `torch.optim.SGD` 类的 `optimizer.state` 在未启动动量时不保存任何状态，`optimizer.state[param]` 为空字典 `{}`，因为标准 SGD 无需历史信息，每步仅依赖当前梯度。

**启用了动量（`momentum > 0`）**时，`optimizer.state[param]` 会包含**`'momentum_buffer'`**：类型为与参数同形状的 `torch.Tensor`。
意义是**动量缓冲（velocity）**，即历史梯度的累积值 $ v_t $ 。 

- 更新规则（忽略 dampening 和 weight_decay）：

  $v_t = \text{momentum} \cdot v_{t-1} + g_t$

- 该缓冲会随着学习过程更新，并用于计算参数更新量。

即使启用了 `weight_decay` 或 `dampening`，**SGD 也不会在 `state` 中额外保存其他状态**，这些操作是即时计算的。

`nesterov` 是一个布尔标志，不影响 `state` 的内容，只影响 `step()` 中的计算逻辑。





### 2.4 优化器的使用方法

PyTorch 中 `torch.optim` 框架中的优化器有非常通用的使用方法。无论使用哪种优化器（如 SGD、Adam、RMSprop 等），基本使用流程都是一致的。

#### 2.4.1 将参数（分组）导入优化器

优化器需要持有自己需要优化训练的参数的引用，不然无法发挥作用，本部分介绍如何将这写参数导入优化器

在目前的几乎全部应用中，**单个网络只会采用一种优化器**，因此这里也只介绍一个网络中用一个优化器的例子，但是相同方法可以方便的迁移到在单一网络中采用多种优化器的情况。

之前已经介绍了创建优化器对象的一般方法，即用构造器：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

其中 `model.parameters()` 能获取pytorch 框架构造的神经网络的全部可学习参数。

如果想对不同层次分别指定优化策略，构造函数也支持，但是在介绍之前，需要先简单介绍一下 pytorch 是如何搭建多层网络的。

在pytorch框架中，由于已经提供了各种网络层次的基础实现，因此只需要在神经网络类的构造函数中直接声明就行

> 但是需要注意的是，构造函数只是声明了网络包含哪些层，实际上并不将这些层搭建成网络

```python
def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(784, 512)
    self.layer2 = nn.Linear(512, 256)
    self.layer3 = nn.Linear(256, 10)
```



而 pytorch 的优化器类实现，就支持通过列表的方式，分别引用不同层次的网络的参数并为其设置优化器超参数（即优化器的构造器参数）

```python
# 为每一层设置不同的学习率
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-4},
    {'params': model.layer2.parameters(), 'lr': 5e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-3}
])
```



而相关的超参数变量会被存放在优化器对象的 `param_groups` 变量中。参考结构如下

```python
optimizer.param_groups = [
    {
        'params': [tensor1, tensor2, ...],   # 参数列表
        'lr': 1e-3,                          # 学习率
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0,
        'amsgrad': False
    },
    # 可能有多个组（如分层学习率）
]
```



#### 2.4.2 在训练中的使用方法

优化器实际上不必须作为模型类型的成员变量，所以这里只介绍优化器如何在训练过程中发挥作用

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 清零梯度
        optimizer.zero_grad()
        
        # 2. 前向传播
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        
        # 3. 反向传播
        loss.backward()
        
        # 4. 更新参数
        optimizer.step()
```

其中，**梯度清零 (`zero_grad()`)** 方法用于清空暂存的梯度，因为**pytorch 采用的是累计梯度的方法**，如果不清零的话上一 batch 的数据会累积到新batch中

- 也可以使用 `model.zero_grad()`，但推荐使用 `optimizer.zero_grad()`

然后，通过调用优化器的 `step()` 方法执行参数更新

> 即，`backward` 方法仅反向传播，计算参数对损失函数的梯度，而不更新参数



#### 2.4.3 优化器状态参数的获取、保存与加载

如上所述，优化器持有自身状态张量 `state` 以及超参数张量 `params` ，需要时可以将其暂存，这对于**断点续训、模型复现、分布式训练**等场景非常重要。

优化器类提供了 `state_dict()` 方法，包揽式的返回优化器需要保存的所有状态。

以及 `load_state_dict` 方法，可以直接从状态字典重新构建相同状态的优化器，需要**注意 `load_state_dict()` 方法并非静态的**，意味着开发者要先创建优化器对象，再调用这个对象的 `load_state_dict` 方法。初始参数参数**会被后续的 `load_state_dict()` 覆盖**，所以即使你记错了原始参数，只要后续正确加载 `state_dict`，最终状态仍然是正确的。 

保存和加载优化器的状态的示例代码如下

```python
# 完整保存
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # 包含state和param_groups
}
torch.save(checkpoint, 'good_checkpoint.pth')

# 完整加载
loaded_checkpoint = torch.load('good_checkpoint.pth')
model.load_state_dict(loaded_checkpoint['model_state_dict'])
optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])  # 同时恢复state和param_groups
```



### 2.5 学习率调度器

学习率调度器（learning rate scheduler）是一种在训练过程中调整学习率的方法，通常会随着训练的进展而降低学习率。这有助于模型在训练开始时当参数远离其最佳值时进行大量更新，并在稍后当参数更接近其最佳值时进行较小的更新，从而允许进行更多微调。

在目前深度学习学科中的共识是：在深度学习的实践中，**“优化器 + 学习率调度器” 搭配使用通常比单纯使用优化器的效果更好**

而在pytorch框架中，默认调度器就是要配合优化器使用

#### 2.5.1 常见调度方法

**阶梯衰减**：阶梯衰减（step decay）调度器每隔几个时期将学习率降低一个常数因子。它的形式定义为：
$$
\text{lr}_t = \text{lr}_0 \times \gamma^{\left\lfloor \frac{t}{\text{step\_size}} \right\rfloor}
$$
其中：

- $ \text{lr}_0 $：初始学习率  
- $ \gamma $：衰减因子（如 0.1）  
- $ t $：当前 epoch  
- $ \text{step\_size} $：每隔多少 epoch 衰减一次

简单、直观、计算开销小,常用于图像分类任务（如 ResNet 在 ImageNet 上常用每 30 epoch 衰减 10 倍）.

但是它是突变性调整，可能在以下情况中导致震荡：

- 如果在衰减点前，优化器正以较大步幅在“稳定震荡”中前进，那么衰减后，它的更新方向和惯性突然不匹配，导致在新的小步长下“重新找平衡”。
- 如 SGD with momentum 或 Adam，动量项保留了之前较大学习率下的历史梯度信息；学习率突降会让动量与当前步长不协调，可能出现短期震荡。

pytorch 中 `torch.optim.lr_scheduler.StepLR` 类实现了阶梯下降调度器，其构造函数为

```python
torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size,
    gamma=0.1,
    last_epoch=-1,
    verbose=False
)
```

参数说明：

- `optimizer` (`torch.optim.Optimizer`)：要调整学习率的优化器。
- `step_size` (`int`)：每隔多少个 epoch 调整一次学习率。
- `gamma` (`float`, 可选)：学习率衰减的乘数因子，默认为 `0.1`。
- `last_epoch` (`int`, 可选)：上一个 epoch 的索引，默认为 `-1`。用于恢复训练时的状态。
- `verbose` (`bool`, 可选)：是否在每次更新学习率时打印信息，默认为 `False`（PyTorch 1.9+ 支持该参数）。



**指数衰减**：Exponential Decay（指数衰减）每个 epoch 都将学习率乘以一个固定的衰减率，使学习率呈指数下降。

它的学习率变化公式为：
$$
\text{lr}_t = \text{lr}_0 \times \gamma^t
$$

- $ \gamma \in (0, 1) $：衰减率（如 0.95）

指数衰减的特点是：学习率平滑、连续地下降，初期下降较快，后期非常缓慢，因此**适合希望模型在后期进行精细微调的场景**。

但是可能衰减过快，导致后期学习能力不足。

`torch.optim.lr_scheduler.ExponentialLR` 类实现了指数衰减学习率调度器，其构造函数签名如下：

```python
torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma,
    last_epoch=-1,
    verbose=False
)
```

参数说明：

- `gamma` (`float`)：每轮（epoch）学习率衰减的乘数因子（即新学习率 = 旧学习率 × gamma）。

  

**余弦退火**：Cosine Annealing（余弦退火）将学习率按照余弦函数从初始值平滑地衰减到最小值（通常接近 0）。其灵感来自模拟退火思想，强调“平滑收敛”。

但是如果只是在一定 epoch 内将学习率衰减到很小，没法满足不同训练时的情况，因此在实践中通常结合热重启。**热重启就是周期性地将学习率“重置”回较高值（即“重启”），然后再次按余弦规律衰减**

这种“热重启”（Warm Restarts）模拟了优化过程中多次探索-精调的循环，既保留了收敛稳定性，又增强了跳出不良局部最优的能力。

它在第 $ t $ 个epoch学习率的表示公式为：
$$
\text{lr}_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{T_{\text{cur}} \pi}{T_i}\right)\right)
$$

其中：

- $ \eta_{\max} $：当前周期的初始学习率（通常逐周期衰减）；
- $ \eta_{\min} $：最小学习率（如 0）；
- $ T_i $：第 $ i $ 个周期的长度；
- $ T_{\text{cur}} $：当前周期内已进行的 epoch 数（从 0 开始计数）；
- 每当 $ T_{\text{cur}} = T_i $ 时，触发一次重启，$ T_{\text{cur}} $ 归零，并可能更新 $ T_i $ 和 $ \eta_{\max} $。



在实践中通常也会对功能进行一些扩充：

- **周期长度的扩展（可选）：**
  - 固定周期：所有周期长度相同（$ T_i = T_0 $）；
  - 指数增长周期：$ T_i = T_0 \times \text{mult}^i $（如 `mult=2`，则周期长度为 $ T_0, 2T_0, 4T_0, \dots $），让后期探索更精细。
- **学习率上限衰减（可选）：**$ \eta_{\max} $ 也可以在每次重启时衰减（如乘以 0.9），避免后期扰动过大。

在pytorch的官方实现类 `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` 中实现了周期长度的指数变化，但是没有实现学习率上限的衰减。其构造函数签名为：

```python
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0,
    T_mult=1,
    eta_min=0,
    last_epoch=-1,
    verbose=False
)
```

**参数说明：**

- `T_0` (`int`)：第一个重启周期的长度（单位：epoch）。即从初始学习率退火到 `eta_min` 所需的 epoch 数。
- `T_mult` (`int`, 可选)：周期增长因子。每次重启后，周期长度变为 `T_prev * T_mult`。默认为 `1`（即所有周期长度相同）。
- `eta_min` (`float`, 可选)：学习率可达到的最小值。默认为 `0`。

当到达周期终点时，**学习率立即重置回初始值**（即优化器中设置的原始学习率），并开始新一轮余弦退火。



最后介绍一下pytorch为用户自定义调度策略设置的类： `torch.optim.lr_scheduler.LambdaLR` 类。这是一个**完全由用户自定义学习率调整策略**的调度器。你只需提供一个或多个 **lambda 函数（或可调用对象）**，该函数接收当前 epoch（或 step）作为输入，返回一个**缩放因子（乘数）**，调度器会将优化器中每个参数组的初始学习率乘以该因子。

>**每次都是基于 optimizer 创建时的原始学习率（即 `base_lrs`）乘以当前 epoch 对应的缩放因子**，而不是在上一个 epoch 的学习率基础上再乘。
>
>学习率调度器的作用只是基于epoch对学习率进行调整，依据训练情况调整学习率应当为优化器的工作

它的构造器签名为：

```python
torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda,
    last_epoch=-1,
    verbose=False
)
```

**参数说明：**

- `lr_lambda` (`function` 或 `list of functions`)：
  - 如果优化器只有一个参数组，传入一个函数：`lambda epoch: ...`
  - 如果有多个参数组，传入一个函数列表，每个函数对应一个参数组。
  - 函数输入为当前 epoch（从 0 开始计数），输出为一个**缩放因子（float）**。



#### 2.5.2 调度器的使用方法

**学习率调度器的作用不是替代优化器，而是和优化器一并调控学习率**

在调控学习率上，调度器的使用方法和优化器相似，**都是调用对象的 `step()` 方法**，但是根据调度方法的不同，调用的时机也不同

例如 `stepLR` 阶梯衰减调度器是在 epoch 结束后调用

```python
for epoch in range(20):
    # 训练一个 epoch（简化）
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
    
    # 在 epoch 结束后调用
    scheduler.step()
```

而带热重启的余弦退火调度器则是在每个 batch 训练结束后都要调用

```python
for epoch in range(20):
    # 训练一个 epoch（简化）
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
    
    # 在 epoch 结束后调用
    scheduler.step()
```



#### 2.5.3 调度器状态的保存和加载

和优化器相似，学习率调度器也会自动维护目前的训练状态，可以用 `state_dict()` 方法得到以字典形式组织的调度器状态并序列化保存

同样的，读取了学习率调度器的状态后可以通过对象的 `load_state_dict` 方法加载调度器相关参数到对象中

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # 👈 关键！
    'loss': loss,
}

torch.save(checkpoint, 'checkpoint.pth')

#加载
checkpoint = torch.load('checkpoint.pth')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 加载调度器状态（必须在 optimizer 加载之后！）
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1
```



## 3 损失函数

深度学习的优化方法直接作用的对象是损失函数。在最优化、统计学、机器学习和深度学习等领域中经常能用到损失函数。损失函数就是用来表示预测与实际数据之间的差距程度。一个最优化问题的目标是将损失函数最小化，针对分类问题，直观的表现就是分类正确的样本越多越好。在回归问题中，直观的表现就是预测值与实际值误差越小越好。

### 3.1 常见任务类型

损失函数是用于衡量模型输出与真实样本之间的差距的，因此损失函数的选取通常和模型的输出相关，进一步地，如果追根溯源，在设计模型时，是任务的类型决定的模型的输出，所以我们可以说**损失函数选择的根本依据是模型需要完成的任务类型**。

因此本部分给出一些常见的任务类型以及该类型下常见的的损失函数

| 任务类型   | 是否监督      | 典型输出                                 | 常见损失函数                                                 |
| ---------- | ------------- | ---------------------------------------- | ------------------------------------------------------------ |
| 回归       | 是            | 连续值                                   | MSE, L1, SmoothL1                                            |
| 二分类     | 是            | 0/1 或概率                               | BCELoss                                                      |
| 多分类     | 是            | 单一类别标签                             | CrossEntropyLoss                                             |
| 多标签分类 | 是            | 多个独立 0/1 标签                        | BCELoss, SoftMarginLoss                                      |
| 实例分割   | 是            | 每个物体实例都有独立的像素级掩码（mask） | 通常结合 CrossEntropyLoss （分类）、DiceLoss / BCELoss（掩码分割）、SmoothL1Loss（框回归） |
| 图像分割   | 是            | 像素级类别图                             | CrossEntropyLoss, DiceLoss                                   |
| 目标检测   | 是            | 边界框 + 类别                            | SmoothL1 + CrossEntropy                                      |
| 序列标注   | 是            | 序列标签                                 | CrossEntropy (+CRF)                                          |
| 生成任务   | 否/弱监督     | 新数据样本                               | Adversarial, Reconstruction                                  |
| 强化学习   | 否            | 策略/动作                                | Policy Gradient, MSE (critic)                                |
| 对比学习   | 自监督        | 特征表示                                 | InfoNCE, Triplet Loss                                        |
| 异常检测   | 无监督/半监督 | 异常分数                                 | Reconstruction Loss                                          |
| 聚类       | 无监督        | 簇标签                                   | Cluster Assignment Loss                                      |

这些类型在实际项目中常常交叉出现（如“多任务 + 多标签 + 时序”），理解问题本质有助于选择合适的模型和损失函数。

> 例如，上述的实例分割问题本质上就是一个交叉问题，它的任务实际上可以看作是多个层次任务的结合：
>
> - **目标检测**：找出图像中每个物体的**边界框（Bounding Box）** 和 **类别（Class）**
> - **语义分割**：对每个像素打上类别标签
> - **实例分割**：不仅要分割出“这是人”，还要区分“这是第1个人”和“这是第2个人”——即**每个物体实例都有独立的像素级掩码（mask）**
>
> 因此，它的损失函数也是三个损失函数的加权和：Total Loss = L_class + L_box + L_mask



### 3.2 损失函数介绍

#### 3.2.1 均方误差损失函数及其变体

MSE（Mean Squared Error，均方误差）损失函数是机器学习和深度学习中最常用、最基础的回归任务损失函数之一，主要用于衡量模型预测值与真实值之间的平均平方误差。**它通过对所有样本的预测误差平方求平均，来评估模型的性能。误差越小，说明模型预测越准确。**

假设我们有 $ n $ 个样本，真实值为 $ y_i $，模型预测值为 $ \hat{y}_i $（其中 $ i = 1, 2, ..., n $），则 MSE 定义为：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- $ y_i $：第 $ i $ 个样本的真实标签（ground truth）
- $ \hat{y}_i $：第 $ i $ 个样本的模型预测值
- $ n $：样本总数

注意：在深度学习框架（如 PyTorch、TensorFlow）中，有时会使用“批量”（batch）计算，此时 $ n $ 表示当前 batch 的样本数。

MSE 对误差进行**平方**处理，由于平方操作，MSE 对**异常值（outliers）非常敏感**，一个很大的误差会被平方后显著拉高整体损失，因此：

- 误差越大，惩罚越重（因为平方放大了大误差的影响）。
  - 误差为 1 → 贡献 1
  - 误差为 10 → 贡献 100（是前者的 100 倍！）
- 误差为 0 时，损失为 0，表示完美预测。

而且，**MSE 对线性模型是可微且凸函数，这使得它非常适合使用梯度下降法进行优化**。

对单个样本的损失 $ L_i = (y_i - \hat{y}_i)^2 $，其对预测值 $ \hat{y}_i $ 的导数为：

$$
\frac{\partial L_i}{\partial \hat{y}_i} = -2(y_i - \hat{y}_i)
$$

这个梯度形式简单、计算高效，是 MSE 被广泛使用的重要原因之一。



当然，MSE 也有自己的缺点：

- **对异常值敏感**：一个离群点可能导致模型过度拟合该点，影响整体性能。
- **单位问题**：MSE 的单位是目标变量单位的平方（如预测房价，单位是“元²”），不易直观解释。此时常用其平方根——RMSE（Root Mean Squared Error）。
  - RMSE保留了 MSE 对大误差敏感的特性，但单位与原始数据一致。

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$



- **不适用于分类任务**：MSE 是为连续值回归设计的，不适用于类别标签。



pytorch 对MSE 有自己的实现：`torch.nn.MSELoss`，它的构造函数签名如下：

```python
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
```

> 从 PyTorch 1.0 开始，推荐使用 `reduction` 参数，而 `size_average` 和 `reduce` 已被弃用（deprecated），但仍可向后兼容。

- `reduction` 参数指定输出结果的聚合方式，为一个字符串，取值为：
  - `'none'`: 不做聚合，返回逐元素 loss（形状同输入）
  - `'mean'`: 返回所有元素loss的平均值
  - `'sum'`: 返回所有元素的loss和

$$
\text{loss}(x, y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2 \quad \text{(当 } \text{reduction}=\text{'mean'})\\

\text{loss}(x, y) = \sum_{i=1}^{N} (x_i - y_i)^2 \quad \text{(当 } \text{reduction}=\text{'sum'})
$$

其中：

- $ x $ 是预测值（网络输出）
- $ y $ 是真实标签
- $ N $ 是 batch 中的元素总数



而 **RMSE（均方根误差）——虽然很少用于训练，但在报告模型性能时**，由于其单位与样本单位一致， RMSE 往往比 MSE 更直观，因此常作为评估指标出现



#### 3.2.2 平均绝对误差损失函数及其变体

L1损失函数（L1 Loss Function），也称为**平均绝对误差**（Mean Absolute Error, MAE）或**绝对误差损失**，是机器学习和深度学习中常用的一种损失函数，用于衡量模型预测值与真实值之间的差异。由于其通常是对单个样本的预测值和真实值的误差对比的，所以**常用于回归任务**

对于单个样本，L1损失定义为预测值 $\hat{y}$ 与真实值 $ y $ 之间绝对差值：

$$
L_1(y, \hat{y}) = |y - \hat{y}|
$$

对于包含 $n$ 个样本的整个数据集，平均L1损失（即MAE）为：

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

其中：
- $y_i$ 是第 $i$ 个样本的真实标签；
- $ \hat{y}_i $ 是模型对第 $ i $ 个样本的预测值；
- $|\cdot|$ 表示绝对值。

L1损失衡量的是预测值与真实值在**数轴上的距离**（曼哈顿距离的一维形式），因此它对**误差的大小线性敏感**：误差越大，损失越大，且增长是线性的。

与L2损失（均方误差，MSE）不同，L1损失**不会对大误差进行平方放大**，因此对离群值（outliers）更鲁棒。

关于误差函数对预测值的偏导数，通常规定如下：

L1损失函数在 $y = \hat{y}$ 处不可导（因为绝对值函数在0点不可导），但其次梯度（subgradient）存在：

$$
\frac{\partial L_1}{\partial \hat{y}} =
\begin{cases}
-1, & \text{if } \hat{y} < y \\
+1, & \text{if } \hat{y} > y \\
\text{任意值} \in [-1, 1], & \text{if } \hat{y} = y
\end{cases}
$$

在实际优化中（如使用梯度下降），通常在 $\hat{y} = y$ 时取导数为0，或使用次梯度方法处理。

注意：由于导数是分段的常数函数，这可能导致L1损失在：

- **远离最优解时**：没有加速收敛的功能，因为梯度不会随着误差变大而变大。
- **接近最优解时**：没有“缓冲”，梯度不会自动变小，导致参数更新可能出现来回震荡（overshoot），收敛速度慢。

L1 损失函数具有以下**优点**

1. **对异常值鲁棒**：因为误差是线性增长，不像L2那样平方放大，所以单个大误差不会主导整个损失。
2. **解释性强**：MAE的单位与原始数据一致，易于理解和解释（例如：“平均预测偏差为2.3元”）。

其**缺点**如下

1. **在最优解附近不可导**：导致基于梯度的优化算法（如SGD）在接近最小值时可能不稳定或收敛较慢。
2. **对大误差不敏感**：在某些任务中（如需要高精度预测），L1对大误差惩罚不够，可能不如L2（MSE）有效。
3. **非光滑**：不利于使用某些需要二阶导数的优化方法（如牛顿法）。



目前在实践中，应用最多的和L1 损失函数相关的损失函数为 MAE 和 MSE 的一种结合形式**Huber Loss**：
$$
L_\delta(a) =
\begin{cases}
\frac{1}{2}a^2, & |a| \le \delta \\
\delta(|a| - \frac{1}{2}\delta), & |a| > \delta
\end{cases}
$$
其中 ：

- $a = y - \hat{y}$表示误差
- $\delta$ 是阈值超参数。

它在小误差时行为类似L2（可导、梯度平滑），而大误差时行为类似L1（对异常值鲁棒），并且**处处可导**，优化更稳定。并且**通过调节 $\delta$，可在L1和L2之间灵活权衡鲁棒性与平滑性。**

在 pytorch 中，对于 Huber Loss的实现是其在 $\delta$ =1 且缩放后的特例Smooth Loss，其数学表示为：
$$
\text{SmoothL1}(a) =
\begin{cases}
0.5a^2 / \beta, & \text{if } \vert a\vert < \beta \\
\vert a \vert - 0.5\beta, & \text{otherwise}
\end{cases}
$$

其中 $a = y - \hat{y}$，$\beta > 0$ 是平滑阈值（通常设为1）。

在pytorch中的实现类为 `torch.nn.SmoothL1Loss` ，其构造函数签名为：

```python
torch.nn.SmoothL1Loss(
    size_average=None,
    reduce=None,
    reduction='mean',
    beta=1.0
)
```

同理，前两个参数已弃用，**`reduction`**（str, optional, default='mean'）指定对逐元素损失值的聚合方式

Smooth L1 损失函数在计算机视觉（Computer Vision, CV）领域，尤其是在**目标检测**（Object Detection）任务中，扮演着至关重要的角色。它的核心价值在于**对边界框（Bounding Box）回归提供鲁棒且稳定的优化信号**。

在目标检测中，模型不仅要分类（判断“是什么”），还要**回归**（判断“在哪里”）。边界框通常用 4 个参数表示，例如：

- `(x, y, w, h)`：中心坐标 + 宽高；
- 或 `(x1, y1, x2, y2)`：左上角和右下角坐标；
- 更常见的是 **偏移量形式**（如 Faster R-CNN 中的 `(tx, ty, tw, th)`）。

这些回归目标具有以下特点：

1. **可能包含标注噪声或异常值**（如模糊边界、遮挡）；
2. **需要高精度定位**（小误差也要精细优化）；
3. **误差分布不均匀**（大部分预测接近真值，少数偏差很大）。

这正是 **Smooth L1 的理想应用场景**：

- 小误差 → 用 L2（平方）→ 梯度平滑，利于精细调整；
- 大误差 → 用 L1（绝对值）→ 避免梯度爆炸，抑制异常值影响。

这里以 Faster R-CNN 为例，介绍一下具体的使用方式

假设：
- 真实边界框参数：$t = (t_x, t_y, t_w, t_h)$
- 预测边界框参数：$\hat{t} = (\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h)$

则边界框回归损失为：

$$
L_{\text{reg}} = \sum_{i \in \{x, y, w, h\}} \text{SmoothL1}(t_i - \hat{t}_i)
$$


#### 3.2.3 （二元）交叉熵损失函数

在介绍交叉熵损失函数之前，我们需要先理解什么是熵，以及什么是交叉熵

在信息论中，**熵（Entropy）衡量一个随机变量的不确定性**：

- 离散随机变量：$H(X) = -\sum_i p(x_i) \log p(x_i)$
- 连续随机变量：$H(X) = -\int p(x) \log p(x) \, dx$

熵越大，不确定性越高；熵越小，不确定性越低。

而对于信息编码，熵 $H(X)$ 表示随机变量 $X$ 的平均不确定性，也就是**平均每个符号所包含的最小信息量**。如果我们要用二进制去表示这些符号，熵就是**平均码长的理论下界**：
$$
L_{avg} \geq H(X)
$$
这里 $L_{avg}$ 是编码后平均每个符号的比特数。

在实际通信或压缩中，我们希望 **编码长度越短越好**，但同时保证能够 **无歧义解码**。

- 如果编码得太长，效率低；
- 如果编码得太短，可能丢失区分能力。

香农源编码定理（Shannon Source Coding Theorem）告诉我们：对一个信息源 $X$，存在编码方法能使平均码长 **任意接近熵 $H(X)$**，但不能低于它。

>熵通常是一个实数，甚至是小数，而实际的码长必须是整数（比如二进制中至少要有 1 位、2 位、3 位…），因此熵 $H(X)$ 本质上是一个 **平均码长的期望值**，而不是对某个具体符号的整数码长。它告诉我们：**如果符号很多，平均下来每个符号需要的信息量就是 $H(X)$ 比特。**

而交叉熵（Cross Entropy）衡量的是用一个分布 $q$ 来编码来自另一个分布 $p$ 的信息所需的平均比特数。

对于交叉熵“用一个分布去编码另一个分布”，我们可以用天气作为例子来帮助理解：

- 晴天概率：$p(\text{晴}) = 0.5$
- 阴天概率：$p(\text{阴}) = 0.25$
- 雨天概率：$p(\text{雨}) = 0.25$

如果我们知道真实的分布 $p$，我们可以设计最优编码：

- 晴天：编码 "0" (1比特)
- 阴天：编码 "10" (2比特)
- 雨天：编码 "11" (2比特)

平均编码长度：
$$
H(p) = -\sum p(x) \log_2 p(x) = -(0.5 \log_2 0.5 + 0.25 \log_2 0.25 + 0.25 \log_2 0.25) = 1.5 \text{ 比特}
$$

这就是熵 $H(p)$ 的含义：用真实分布 $p$ 编码时的最小平均编码长度。

但在现实中，我们往往**不知道真实的分布 $p$ ，只能用一个估计的分布 $q$** 来设计编码方案。

真实分布 $p$（实际天气规律）：
- 晴天：0.5
- 阴天：0.25
- 雨天：0.25

估计分布 $q$（我们的错误认知）：
- 晴天：0.25
- 阴天：0.25
- 雨天：0.5

基于错误的分布 $q$，我们会设计这样的编码：
- 晴天：编码 "10" (2比特，因为认为概率低)
- 阴天：编码 "11" (2比特)
- 雨天：编码 "0" (1比特，因为认为概率高)

现在我们**计算用基于 $q$ 设计的编码方案，来编码实际来自分布 $p$ 的数据，平均需要多少比特**

- 晴天实际出现概率是 0.5，但我们用了 2 比特编码
- 阴天实际出现概率是 0.25，我们用了 2 比特编码
- 雨天实际出现概率是 0.25，我们用了 1 比特编码

平均编码长度：
$$
H(p, q) = 0.5 \times 2 + 0.25 \times 2 + 0.25 \times 1 = 1.75 \text{ 比特}
$$

用公式表示就是：
$$
H(p, q) = -\sum p(x) \log_2 q(x)
$$

这就是交叉熵的含义：用分布 $q$ 的编码方案来编码来自分布 $p$ 的数据所需的平均比特数。

类比到机器学习中

- **真实分布 $p$ 对应着数据的真实标签分布**：比如对于分类问题，通常是 one-hot 分布（真实类别概率为1，其他为0）；
- **估计分布 $q$ 对应着模型预测的概率分布**：比如神经网络输出的 softmax 概率。

交叉熵损失就是在衡量：用模型预测的分布来“编码”真实标签需要多少“信息量”。

- 当模型预测完全正确时（$q = p$），交叉熵 = 熵，达到最小值
- 当模型预测错误时，交叉熵增大，表示“编码效率”变差



利用熵和交叉熵作为工具，我们还可以构建出**衡量两个概率分布之间的差异的数学量 KL 散度**（KL Divergence）：
$$
D_{KL}(p \parallel q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)} = H(p, q) - H(p)
$$

关系：交叉熵 = 熵 + KL散度



而将交叉熵作为损失函数，就是交叉熵损失函数（Categorical Cross-Entropy）。它**适用于分类间互斥的多分类问题**。

数学定义为：

对于 $C$ 个类别，真实标签用 one-hot 编码表示：

- 真实标签：$\mathbf{y} = [y_1, y_2, ..., y_C]$，其中 $y_i \in \{0,1\}$ 且 $\sum y_i = 1$
- 预测概率：$\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, ..., \hat{y}_C]$，其中 $\hat{y}_i \in [0,1]$ 且 $\sum \hat{y}_i = 1$

损失函数：
$$
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$
而在实际计算中，由于 one-hot 编码的特性，只有一个 $y_i = 1$，其余为 0，所以实际上：
$$
L = -\log(\hat{y}_{\text{true}})
$$

其中 $\hat{y}_{\text{true}}$ 是真实类别对应的预测概率。

而当只有两个类别时，交叉熵损失特化为二元交叉熵：

- 多分类形式：$L = -[y_1 \log(\hat{y}_1) + y_2 \log(\hat{y}_2)]$
- 由于 $y_2 = 1 - y_1$ 且 $\hat{y}_2 = 1 - \hat{y}_1$
- 得到：$L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$

注意到二元交叉熵损失函数只需要一个输出神经元 $p$，它表示“属于正类的概率”，那么负类概率就是 $1-p$，这和Sigmoid 把一个实数压缩到 $[0,1]$ 的输出特性天然兼容，因此：

- 用二元交叉熵损失函数的网络输出层一般都是sigmoid激活函数
- 更多种类互斥分类的网络的输出层一般都是使用softmax激活

此外，**二元交叉熵损失函数也可以用于多标签的二分类**——有多个标签，每个样本可能属于多个类别（对每个类别来说都是二分类）。其方法可以概括为：每个类别独立进行二分类，输出层使用sigmoid激活，最后损失函数对每个类别分别计算BCE然后求和或平均。





此外，对于（二元）交叉熵损失函数，**实际应用中可以作出如下改进**：

为防止模型过于自信，可以使用**平滑标签（Label Smoothing）**，其原理为将真实分类的概率设为接近1的概率 $ \epsilon $ ，而剩下的 $1-\epsilon$ 的可能性被其他概率平分。例如3分类，`ε = 0.1`）：

- 原始硬标签：`[1, 0, 0]`
- 平滑后软标签：`[0.9, 0.05, 0.05]`

采用平滑标签的原因是标准交叉熵损失会驱使模型对正确类别输出概率接近1，对错误类别接近0，使得模型对于相似类型的区分能力下降。而标签平滑相当于告诉模型"真实答案不是100%确定的"，鼓励模型保持适度的不确定性，最终在一定程度上避免了极端概率输出。

而从信息论角度看，标签平滑**增加了目标分布的熵**：`H(ỹ) > H(y)`，**降低了模型需要达到的确定性**，使得模型不需要将输出概率压缩到极端值，从而**提高了编码效率的鲁棒性**，对噪声和异常值更不敏感。



当类别不平衡时，可以使用加权交叉熵：

$$
L = -\sum_{i=1}^{C} w_i y_i \log(\hat{y}_i)
$$

其中 $w_i$ 是类别权重，通常与类别频率成反比。



对于交叉熵损失函数，pytorch提供了开箱即用的 `nn.CrossEntropyLoss` 类，该类在使用时**自动完成了以下两个步骤**：

1. **LogSoftmax**: 将原始logits转换为对数概率
2. **NLLLoss** (Negative Log Likelihood Loss): 计算负对数似然损失

因此**在pytorch中，对于大于两个标签的多分类，如果使用该类**，就不需要在模型最后的输出层用softmax函数处理。

它的构造函数签名为：

```python
torch.nn.CrossEntropyLoss(
    weight=None, 
    size_average=None, 
    ignore_index=-100, 
    reduce=None, 
    reduction='mean', 
    label_smoothing=0.0
)
```

下面给出一些参数的说明：

- **`ignore_index`** (`int`, optional, default=-100)指定一个目标值，该值对应的损失将被忽略（不参与梯度计算）。常用于序列标注任务中忽略填充（padding）位置的损失计算。
- **`label_smoothing`** (`float`, optional, default=0.0)代表标签平滑因子，取值范围为 `[0.0, 1.0]`。当值大于 0 时，会将真实标签从硬标签（one-hot）转换为软标签，有助于防止模型过拟合和提高泛化能力。



同理，在pytorch中也提供了结合了sigmoid函数的开箱即用的二元交叉熵损失函数类 `nn.BCEWithLogitsLoss`，其构造函数签名为：

```python
torch.nn.BCEWithLogitsLoss(
    weight=None, 
    size_average=None, 
    reduce=None, 
    reduction='mean', 
    pos_weight=None
)
```

这个类同样适用于传统二分类以及多标签的二分类。

###  3.3 自定义损失函数



### 3.4 损失函数的使用方法