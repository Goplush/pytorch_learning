



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
H(p) = -\sum p(x) \log_2 p(x) \\= -(0.5 \log_2 0.5 + 0.25 \log_2 0.25 + 0.25 \log_2 0.25) = 1.5 \text{ 比特}
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

最后，可以看出，二元交叉熵，或者说 n 元交叉熵损失函数假设对于对于每一个具体的输入 $x_i$，其真实标签 $y_i$ 服从参数为 $ \mathbf{p}_i = f_\theta(x_i) $ 的 $n$ 项分布（Multinomial Distribution），这也是模型的输出不是一个给定的标签，而是 $n$ 个logist得分。

上述说法的**数学表示**为：
$$
y_i | x_i \sim \text{Multinomial}(1, \mathbf{p}_i) = \text{Multinomial}(1, f_\theta(x_i))
$$

其中：

- $ y_i \in \{0,1\}^K $ 是 one-hot 编码的真实标签  
- $ \mathbf{p}_i = [p_{i1}, p_{i2}, \dots, p_{iK}] \in [0,1]^K $ 是模型预测的概率分布  
- $ \sum_{k=1}^{K} p_{ik} = 1 $  
- $ f_\theta(\cdot) $ 通常是带有 softmax 激活的神经网络





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

对于传统二分类，即要预测 **样本是否属于某一类**（例如“是否为猫”）：

- **输入 (`input`)：**
  - 形状：`(N,)` 或 `(N, 1)`
  - 含义：网络对每个样本输出的 **logit 值**（未经过 sigmoid 的实数，可以正可以负）。
- **目标 (`target`)：**
  - 形状：`(N,)` 或 `(N, 1)`
  - 含义：每个样本的真实标签，取值为 `0` 或 `1`。

而对于多标签二分类，即一个样本可以同时属于 **多个类别**，例如：一张图片既是“猫”又是“可爱”，标签是多热（multi-hot）向量。

- **输入 (`input`)：**
  - 形状：`(N, C)`
  - 含义：网络对每个样本的每个类别的 logit 值。
    - `N` = batch size
    - `C` = 类别数（每个类别都是独立的二分类）
- **目标 (`target`)：**
  - 形状：`(N, C)`
  - 含义：真实标签的 multi-hot 向量，元素为 `0` 或 `1`。

而 `weight` 和 `pos_weight` 两个参数则是分别用于缓解以上两种情况的类别不平衡的：

`weight` 参数常用于多标签的类别权重平衡，它的作用是对 **每个元素的 loss** 乘以一个权重，因此**形状要求**为`(N, C)` 或可以广播成 `(N, C)`。这个的作用原理比较好理解，就不多赘述了

- 通常来说，每个标签要乘的权重是相同的，如果真的要为每个样本都单独设计权重，那工作量就太大了

而 `pos_weight` 则主要用于传统二分类的类别权重平衡——比如在**医疗诊断**中，**健康患者占比 99%，患病患者占 1%，如果采用标准的二元交叉熵损失函数，模型很可能会学会"总是预测负类"也能获得较低的总体损失**。因此**可以为样本量少的标签设置大于一的权值**，来让模型更多的学习来自这个标签的损失：

当设置 $\text{pos\_weight} = \alpha$ 时，损失函数变为：
$$
L = -[\alpha \cdot y \cdot \log(\sigma(x)) + (1 - y) \cdot \log(1 - \sigma(x))]
$$
其中：
- $x$ : logits（模型输出）
- $y$ : 真实标签（0 或 1）
- $\sigma(x)$ : sigmoid 函数

注意到正样本（$y=1$）的损失被放大了 $\alpha$ 倍，而负样本（$y=0$）的损失保持不变——这也是**pytorch框架识别少数样本的方式：依赖用户将少数样本标记为正样本`1`。**

`pos_ weight` 参数在初始值的设置上通常参考正负样本的比例：`pos_weight = num_neg / num_pos` ，在训练或者测试中，也可以根据相关指标调整该参数：

- 如果召回率太低，适当增大 pos_weight
- 如果精确率太低，适当减小 pos_weight

多标签二分类也能使用 `pos_weight`参数，只需要为每个标签都设置正样本权重即可（`pos_weight` 为一个**长度等于类别数量**的向量，其中**每个位置对应一个类别的正样本权重**。）



#### 3.2.4 Hinge Loss 与 SoftMargin Loss

首先回顾一下标准二元交叉熵损失函数**（标签在 {0,1}）**，对于样本  $(x,y )$
$$
\mathcal{L}_{\text{BCE}} = -\left[ y \log(\sigma(f) + (1-y)\log(1-\sigma(f)) \right]
$$
其中

- $ f $ : 等价于 $ f(x) $，代表着logits（模型输出）
- $ y $ : 真实标签（0 或 1）
- $\sigma(f)$ : sigmoid 函数

而设新的标签 $y' \in \{-1, +1\}$，并且：
$$
y = \frac{y' + 1}{2}  \quad \Leftrightarrow \quad y' = 2y - 1
$$
代入 BCE，整理后确实能得到：
$$
\mathcal{L}' = \log \left(1 + \exp(-y' f)\right)
$$
而PyTorch 的 `nn.SoftMarginLoss` 定义就是：
$$
\mathcal{L}_{\text{SoftMargin}}(f,y') = \frac{1}{N} \sum_{i=1}^N \log(1 + \exp(-y'_i f_i))
$$
就是上面 $\mathcal{L}'$ 的批次形式，因此——

**SoftMarginLoss 本质上就是 Binary Cross-Entropy 损失在标签 $\{-1,+1\}$ 情况下的等价形式**。因此二者在使用上的效果基本也是一致的

（本部分要介绍的主要内容到此就结束了，下面是关于 SoftMargin loss 来源的说明，顺便补充对于 SVM，支持向量机，有关的内容，不重要，可以跳过）

---

其实soft margin 损失函数不是对于二元交叉熵损失函数的巧妙模仿，而是对另一个函数的有效改进——这就是我们标题在的另一个损失函数 Hinge Loss。

Hinge Loss 是支持向量机（SVM）中常用的损失函数，**用于二分类任务**。它的目标是最大化分类间隔（margin），从而提高模型的泛化能力。

对于一个样本 $(x_i, y_i)$，其中：
- $y_i \in \{-1, +1\}$ 是真实标签，
- $f(x_i)$ 是模型的原始输出（未经过 sigmoid 或 softmax 的“得分”或“logit”），

Hinge Loss 的定义为：
$$
\mathcal{L}_{\text{hinge}} = \max(0, 1 - y_i \cdot f(x_i))
$$

直观理解：
- 如果 $y_i f(x_i) \geq 1$，说明样本被正确分类且置信度足够高（在“安全边界”之外），损失为 0。
- 如果 $y_i f(x_i) < 1$，说明样本要么分类错误，要么虽分类正确但置信度不够（在边界内），此时会产生正的损失。

Hinge Loss 的特点是不可导（在 $y_i f(x_i) = 1$ 处不可导），且对正确分类但靠近边界的样本也会惩罚。

为了解释 Hinge Loss 最大化分类间隔的优化原理，我们着重看 $y_i\cdot f(x_i)$ 的部分，把它称之为函数间隔，记为 $\hat\gamma_i$

如果分类正确，$\hat\gamma_i=y_i f(x_i) \gt 0$，这里依赖于二分类 $y_i \in \{-1, +1\}$ 的规定：如果标签 $y_i$ 为正数，预测器应当输出正数，反之亦然。

而且函数间隔 $\hat{\gamma}_i = y_i f(x_i)$ 反映了**离决策边界的距离**（在权重未归一化的情况下是比例距离），它越大，意味着 $x_i$ 离分隔超曲面越远，因此分类器“更确信”该点属于其类别。

因此， **Hinge Loss 实际上是在惩罚函数间隔小于 1 的样本**。

但是函数间隔会受到函数本身缩放的影响，这里举个简单的例子：对于一个连续的预测函数 $ f $，将其放大1e100倍，即 $ f' = 10^{100}f $，此时所有的输出都显著更加远离了决策边界，但是实际上的决策边界仍然没有改变，这使得需要引入额外的限制才能保证训练结果的有效性。

再加上hinge loss 只给了一个几何 margin 的间隔解释，**没有概率意义**。而logistic / cross-entropy 可以自然接在 sigmoid/softmax 之后，直接输出“概率分布”，这对分类、检测、语言建模等任务都至关重要。

以及 hinge loss 在 margin $y f(x) = 1$ 的地方有**不可导点**。使得反向传播稳定性不足。这些缺点使得 hinge loss基本只出现在SVM中

在 SVM 中，预测函数为一个线性函数
$$
f(x_i)=w^\top x_i + b
$$
其中 $\omega$ 为可学习的权重向量，$b$ 为可学习的偏置。

那 SVM 预测函数的函数间隔就是
$$
\hat{\gamma}_i = y_i (w^\top x_i + b)
$$
此时，在决策边界固定的情况下，可以方便的通过约束 $\|\omega\|$ 来约束函数的缩放对输出的影响，参考几何距离 $ \gamma_i $​ 可以方便的证明这一点
$$
\gamma_i = \frac{y_i (w^\top x_i + b)}{\|w\|} = \frac{\hat{\gamma}_i}{\|w\|}
$$
下面给出 SVM 的标准损失函数（优化问题）：
$$
\min_{w,b} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i (w^\top x_i + b)\right)
$$
这个目标函数包含两部分：

1. 正则项 $\frac{1}{2} \|w\|^2$：
   - 控制模型复杂度；
   - **等价于最大化几何间隔**：因为几何间隔 $\gamma = \frac{1}{\|w\|}$（当函数间隔被约束为 $\geq 1$ 时），所以最小化 $\|w\|$ 就是最大化 $\gamma$。
   
2. Hinge Loss 项 $\sum \max(0, 1 - y_i f(x_i))$：
   - 惩罚那些函数间隔 $< 1$ 的样本（包括错分和靠近边界的样本）；
   - 允许一些样本违反间隔约束（通过参数 $C$ 控制容忍度），即“软间隔 SVM”。

我们可以看出，损失函数的第一部分已经做到了最大化几何间隔，下面介绍第二部分的作用

对于现实情况下数据存在噪声，而且决策分界往往本身非线性的情况：SVM 通过引入 松弛变量（slack variables）$\xi_i \geq 0$ 来放松约束：
$$
y_i (w^\top x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

- 如果 $\xi_i = 0$：样本在正确一侧且在间隔外（理想情况）
- 如果 $0 < \xi_i < 1$：样本在间隔内，但分类正确
- 如果 $\xi_i \geq 1$：样本被错误分类

因此，优化目标可以表示为：
$$
\min_{w,b,\xi} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

其中：
- $\frac{1}{2} \|w\|^2$：仍然鼓励大间隔（即小 $\|w\|$）
- $\sum \xi_i$：代表总分类误差（由 Hinge Loss 等价表达）
- $C > 0$：正则化参数，控制“间隔最大化”和“分类准确率”之间的权衡

上面的优化目标就是
$$
\min_{w,b} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i (w^\top x_i + b)\right)
$$


#### 3.2.5 Dice Loss

Dice Loss 源于 **Dice 系数**（Dice Coefficient），也称为 **Sørensen-Dice 系数**，最初用于衡量两个集合的相似度。Dice Loss **主要应用于图像分割**，在医学图像分割领域，Dice Loss 因其对类别不平衡问题的良好处理能力而广受欢迎。

> 图像分割任务的本质，实际上还是判断某个像素点是否属于

对于两个集合 $A$ 和 $B$，Dice 系数定义为：
$$
DSC = \frac{2|A \cap B|}{|A| + |B|}
$$

其中：
- $|A \cap B|$ 是两个集合的交集大小
- $|A|$ 和 $|B|$ 分别是两个集合的大小

将其应用在图像分割中：
- $A$ 表示预测的分割结果
- $B$ 表示真实的标签（ground truth）
- Dice 系数范围：$[0, 1]$，值越大表示分割效果越好

Dice Loss 是 Dice 系数的补集，其一般形式为：
$$
\text{Dice Loss} = 1 - DSC = 1 - \frac{2|A \cap B|}{|A| + |B|}
$$
对于二值分割，设：
- $p_i$ 为第 $i$ 个像素的预测概率（0-1之间）
- $g_i$ 为第 $ i $ 个像素的真实标签（0 或 1）

Dice Loss通常表示为：
$$
\text{Dice Loss} = 1 - \frac{2 \sum_{i=1}^{N} p_i g_i + \epsilon}{\sum_{i=1}^{N} p_i^2 + \sum_{i=1}^{N} g_i^2 + \epsilon}
$$
其中 $ \epsilon $ 为为了避免分母为零的情况加入的平滑因子，通常取 $10^{-5}$ 或 $ 10^{-8} $

对于多类别分割（$C$ 个类别），有以下几种处理方式：

首先是直接求平均值：
$$
\text{Dice Loss} = \frac{1}{C} \sum_{c=1}^{C} \left(1 - \frac{2 \sum_{i=1}^{N} p_{i,c} g_{i,c} + \epsilon}{\sum_{i=1}^{N} p_{i,c}^2 + \sum_{i=1}^{N} g_{i,c}^2 + \epsilon}\right)
$$
对于类别不平衡，也可以为不同类别分配不同权重 $w_c$：
$$
\text{Dice Loss} = \sum_{c=1}^{C} w_c \left(1 - \frac{2 \sum_{i=1}^{N} p_{i,c} g_{i,c} + \epsilon}{\sum_{i=1}^{N} p_{i,c}^2 + \sum_{i=1}^{N} g_{i,c}^2 + \epsilon}\right)
$$
Dice Loss的核心优势为对不平衡的小目标有更好的效果，这种优势来源于其分母为两个集合的大小之和，下面从 loss 本身和梯度两方面分析

首先，对于极端小的目标，Dice Loss 在学习初期能让模型更好的学习到小目标的识别

举个例子，假设真实小目标只有 3 个像素：

- **完全漏检**时，$\text{Dice Loss} = 1 - 0 = 1.0$；
- **检测到2个像素（漏掉1个）**时， $\text{Dice Loss} = 1 - \frac{2 \times 2}{2 + 3} = 1 - \frac{4}{5} = 0.2$；
- **检测到4个像素（多识别1个）**时，$\text{Dice Loss} = 1 - \frac{2 \times 3}{4 + 3} = 1 - \frac{6}{7} = \frac{1}{7}$

可见对于小目标的存在性检测上， Dice Loss 鼓励模型优先识别到小目标，在识别到的基础上进行边界的优化。这种对存在性识别的鼓励本质上来源于其梯度特性，对于预测概率 $p_i$，Dice Loss 的梯度为：
$$
\frac{\partial \text{Dice Loss}}{\partial p_i} = -\frac{2g_i (|A| + |B|) - 2|A \cap B| \cdot 2p_i}{(|A| + |B|)^2}
$$

当 $|B|$ 很小时（小目标），分母 $(|A| + |B|)^2$ 也很小，导致梯度幅值更大。



#### 3.2.6 对抗损失函数

Adversarial 损失函数（对抗损失函数）是生成对抗网络（Generative Adversarial Networks, GANs）中的核心组成部分，由 Ian Goodfellow 等人在 2014 年首次提出。它的设计思想来源于博弈论中的“二人零和博弈”：一个生成器（Generator）试图生成逼真的假数据，而一个判别器（Discriminator）则试图区分真实数据和生成的假数据。两者在训练过程中相互对抗、共同进化。

它的基本思想是就是 GAN 本身的运行逻辑：

GAN 包含两个神经网络：

- **生成器 G(z)**：接收一个随机噪声向量 $ z $（通常从标准正态分布或均匀分布中采样），输出一个假样本 $ G(z) $，试图模仿真实数据分布。

- **判别器 D(x)**：接收一个样本 $ x $（可以是真实数据或生成器生成的假数据），输出一个标量，表示该样本为“真实”的概率（通常在 [0,1] 区间）。

对抗损失函数的目标是：

- **判别器 D**：最大化其正确分类真实样本和假样本的能力。
- **生成器 G**：最小化判别器识别其生成样本为“假”的能力（即“欺骗”判别器）。




最经典的对抗损失函数是 Minmax Loss，它直观地反映了 GAN 网络的需求：
$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]
$$
其中：

- $ G $ 是生成器（Generator），输入随机噪声 $ z \sim p_z(z) $（如标准正态分布），输出假样本 $ G(z) $。
- $ D $ 是判别器（Discriminator），输入真实样本 $x$ 或生成样本 $ G(z) $，输出一个概率值（表示输入是真实数据的概率）。
- $ p_{\text{data}}(x) $ 是真实数据的分布。
- $ p_z(z) $ 是噪声的先验分布（通常为高斯或均匀分布）。
- $ V(D, G) $ 是判别器和生成器的价值函数（value function）。

这个目标函数是一个极小极大问题：

- **内层（$ \max_D $）**：固定生成器 $ G $，训练判别器 $ D $ 使其尽可能区分真实样本和生成样本（即最大化 $ V(D, G) $），也即对判别器 $ D $ 而言：
  - $\mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)]$ 尽量大说明判别器对真实样本输出尽量高
  - $\mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]$ 尽量大说明判别器对伪造样本输出尽量低
- **外层（$ \min_G $）**：固定判别器 $ D $，训练生成器 $ G $ 使其生成的样本尽可能“欺骗”判别器（即最小化 $ V(D, G) $）。



在实践中，当生成器刚开始训练时，$ D(G(z)) $ 接近 0，导致 $ \log(1 - D(G(z))) $ 的梯度非常小（梯度消失），训练困难。

因此，实践中通常采用**非饱和版本**的生成器损失：

- **判别器损失（不变）**：

$$
\mathcal{L}_D = -\mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] - \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]
$$

- **生成器损失（改为最大化 $ \log D(G(z)) $）**：

$$
\mathcal{L}_G = -\mathbb{E}_{z \sim p_z} [\log D(G(z))]
$$

这样，当 D 认为 G(z) 是假的（即 $ D(G(z)) \approx 0 $）时，$ \log D(G(z)) $ 会趋向负无穷，梯度较大，有利于 G 的早期训练。



对于对抗损失函数，在实践中最常用的就是上述的非饱和的 MinMax 损失函数；



此外，之前介绍的 Hinge Loss 也在对抗模型中有广泛的应用：

 - **判别器损失：**

$$
  \mathcal{L}_D = \mathbb{E}[\max(0, 1 - D(x))] + \mathbb{E}[\max(0, 1 + D(G(z)))]
$$

  - **生成器损失：**

$$
\mathcal{L}_G = -\mathbb{E}[D(G(z))]
$$

对抗的 Hinge Loss 的优势主要体现在工程上：

- 在高分辨率图像生成中表现优异；
  - 被 BigGAN（2019）、StyleGAN / StyleGAN2 / StyleGAN3（NVIDIA）等 SOTA 模型广泛采用；
  - 梯度更“平滑”，有助于稳定训练大规模生成器；
  - 不强制输出概率（D 可输出任意实数），更灵活。



最后介绍一下 GAN 交替进行的训练模式：

1. **更新判别器 D（固定 G）**：

   - 从真实数据中采样一批 $ x $，计算损失：$ -\log D(x) $
   - 从噪声中采样一批 $ z $，生成假样本 $ G(z) $，计算损失：$ -\log(1 - D(G(z))) $
   - 总判别器损失：$ L_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))] $
   - 最小化 $ L_D $（等价于最大化 $ V(D, G) $）

2. **更新生成器 G（固定 D）**：
   - 从噪声中采样 $ z $，生成 $ G(z) $
   - 生成器损失：$ L_G = -\mathbb{E}[\log D(G(z))] $ （注意：这里使用了非饱和的 Min-Max 形式）
   - 最小化 $ L_G $

而 GAN 的理论最优则是依据纳什均衡得到：

- 判别器无法区分真假：$ D^*(x) = \frac{1}{2} $ 对所有 $x$
- 生成器完美模仿真实分布：$ p_g(x) = p_{\text{data}}(x) $

此时价值函数达到全局最优值：

$$
V(D^*, G^*) = \log(1/2) + \log(1/2) = -2 \log 2 \approx -1.386
$$

这说明 Min-Max 损失在理论上可以引导生成器学习到真实数据分布。





#### 3.2.7 重构损失函数

重构损失（Reconstruction Loss）是深度学习中**一类用于衡量模型重建输出与原始输入之间差异的损失函数**。它主要用于自编码器（Autoencoder）、生成模型等架构中。

“重构”的核心思想是：模型接收输入数据后通过编码器压缩为低维表示（**潜在空间表示**），然后通过解码器重建原始输入，最后**重构损失衡量重建结果与原始输入的相似程度**。

重构损失函数的一般表示为：
$$
\mathcal{L}_{\text{recon}} = D(x, \hat{x})
$$
其中：

- 原始输入：$ x \in \mathbb{R}^n $
- 重建输出：$ \hat{x} = g(f(x)) \in \mathbb{R}^n $
- $ f(\cdot) $ 是编码器，$ g(\cdot) $ 是解码器
- $ D(\cdot, \cdot) $ 是某种距离度量函数（如 MSE、L1、SSIM 等），用于衡量原始输入与重建输出之间的差异。

常见的重构损失函数就是本部分开头所讲的：

- 均方误差损失函数，也被称为 L2 损失函数
- 平均绝对误差损失函数，也被称为 L1 损失函数
- 二元交叉熵损失函数

其中 L2、L1 指的是向量的 **"Lp范数"（Lp Norm）**，具体而言，是**模型输出向量对真实向量的损失向量的范数**，数学表示为：

- **L1 损失**：  $\mathcal{L}_1 = \| \mathbf{y} - \hat{\mathbf{y}} \|_1 = \sum_i |y_i - \hat{y}_i|$
- **L2 损失**： $ \mathcal{L}_2 = \| \mathbf{y} - \hat{\mathbf{y}} \|_2^2 = \sum_i (y_i - \hat{y}_i)^2$



对于重构损失函数，在应用上通常不要求模型读取输入后重构整个输入，而是读取被随机掩码的样本，在隐空间内预测被打码的信息，主要原因有以下两点：

- 完整重建的要求使得模型可能学习到"恒等映射"，缺乏泛化能力
- 而掩码重建：
  - 降低了学习时的计算量
  - 强制模型学习数据的内在结构和上下文关系

对于掩码训练，主要有以下几种模式：

- 去噪自编码器（Denoising Autoencoder, DAE）

  ```python
  # 输入添加噪声
  x_noisy = x + noise
  # 或者随机置零（类似dropout）
  mask = torch.rand_like(x) > corruption_rate
  x_corrupted = x * mask
  
  # 训练目标：从损坏数据重建原始数据
  loss = reconstruction_loss(x, decoder(encoder(x_corrupted)))
  ```

- 掩码自编码器（Masked Autoencoder, MAE），这是近年来非常流行的方法，常用于**Vision Transformer（ViT）**的训练中：

  ```python
  class MaskedAutoencoder:
      def __init__(self, mask_ratio=0.75):
          self.mask_ratio = mask_ratio
      
      def forward(self, x):
          # 1. 将图像分块
          patches = self.patchify(x)  # [B, N, D]
          
          # 2. 随机掩码大部分patch
          ids_shuffle, ids_restore = self.random_masking(patches, self.mask_ratio)
          visible_patches = patches.gather(1, ids_shuffle[:, :num_visible])
          
          # 3. 只对可见patch进行编码
          encoded = self.encoder(visible_patches)
          
          # 4. 解码器重建所有patch（包括被掩码的）
          decoded = self.decoder(encoded, ids_restore)
          
          # 5. 只计算被掩码部分的重构损失
          loss = self.reconstruction_loss(
              target_patches[:, masked_indices], 
              decoded[:, masked_indices]
          )
          return loss
  ```

  - 它的关键特点有：
    - **高掩码率**：通常掩码75%的数据
    - **仅计算掩码部分损失**：提高训练效率
    - **不对称架构**：编码器轻量，解码器重

- Bert 风格的掩码（**MLM 任务**）

  - 主要应用于 NLP 领域：
    - 输入序列: [CLS] The cat [MASK] on the mat [SEP]
    - 目标：预测被[MASK]位置的原始词"sat"
    - 损失：只计算被掩码位置的交叉熵损失
  - 它主要应用于 Bert 模型的训练与继续训练（利用特定任务或特定领域的语料对预训练模型进行继续训练）



###  3.3 自定义损失函数

介绍自定义损失函数前，我们需要先了解 pytorch 的自动求导方法，这个我单独找了一个笔记 `ch3/extra/autograd.md`，简单来说就是在上一章中介绍过的：将运算以图的形式组织起来

<img src="D:\data\repos\pytorch_learning\ch3\assets\image-20251003203941039.png" alt="image-20251003203941039" style="zoom:50%;" />

在计算损失函数值的时候是沿着边的指向正向传播（`forward`），而在计算梯度的时候则是反着边的指向传播（`backward`）。

因此，自定义损失函数时，需要着重定义的也就是这两个方法。

下面介绍两种常见的自定义损失函数的方式

#### 3.3.1 继承 nn.module 类

这种方法主要用于将**可以在运算图中追踪梯度的运算**组合成损失函数。这包括了 pytorch 框架提供的全部基础运算、全部开箱即用的损失函数类等等。

符合条件的操作，也被称为是 **pytorch 可微分的**，可以通过下面的代码判断某个操作是否是 pytorch 可微分的

```python
def torch_autograd_test():
    # 测试操作是否可微分

    def test_operation(x:torch.Tensor):
        #该方法为需要测试是否是pytorch可微分的方法
        #the operation to be tested
        return x.add(x)

    x = torch.randn(3, 3, requires_grad=True)
    try:
        y = test_operation(x)
        loss = y.sum()
        loss.backward()
        print("可微分！梯度存在:", x.grad is not None)
    except:
        print("不可微分！")
```

在该例子中，由于仅使用了 pytorch 本身的操作，所以操作是可微的，因此输出

```
可微分！梯度存在: True
```



定义的一般步骤为：

1. 在构造函数中设置超参数、检查参数有效性等
2. 定义 `forward` 方法，通常至少包括 `output` 和 `target` 两个参数，前者指模型网络的输出，后者指期望的输出（真实数据，比如真实标签、被mask的文本、真实图像……）

因为 pytorch 已知每个操作的梯度规则，所以 pytroch 可以**完全自动微分**，无需手动实现 `backward` 方法。



这里给一个自定义将多个已知损失函数加权求和的损失函数作为例子（假设已知的损失函数都是pytorch可微分的）：

```python
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用PyTorch内置损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target, anchor, pos, neg):
        loss1 = self.mse_loss(pred, target)
        loss2 = self.l1_loss(pred, target) 
        
        return 0.5 * loss1 + 0.5 * loss2
```



#### 3.3.2 继承 Function 类（创建自定义 Function）

> 本部分的梯度和偏导可以视为等同概念，但是仅限本部分

由于 pytorch 并不是专门的计算库，一些复杂的运算可能没法方便的组合出来，或者即使可以，编程的难度也会很高。因此可以通过创建自定义的 Function 类来引入非 pytorch 的操作。（通常情况下都是转化为最基础最常用的矩阵运算库numpy的 nparray 类型进行操作）

由于转化成numpy数组的操作以及numpy运算并不是pytorch可微分的，所以在这种方法中pytorch框架无法自动追踪梯度从而实现自动求导，因此继承 Function 类后构建自定义损失函数的一般方法为：

1. 在构造函数中设置超参数、检查参数有效性等
2. 定义**静态的 `forward` 方法**
3. 定义**静态的 `backward` 方法**，至少包含 `grad_output` 参数：
   - 功能是作为链式求导的上游梯度，即 $\frac{\partial \text{loss}}{\partial \text{input}} = \left( \frac{\partial \text{loss}}{\partial \text{output}} \right) \times \left( \frac{\partial \text{output}}{\partial \text{input}} \right)$ 中的前者
   - 形状上和 `forward` 方法的输出一致
   - 当 `forward` 方法有多个输出时，`backward` 方法也有多个输入，分别计算每个输出对输入的梯度后和 `grad_output` 相乘、求和作为梯度输出

此处特地要求 `forward` 和 `backward` 方法是静态的是为了能直接引入 `ctx` 参数，`ctx` 是pytorch 框架为每个运算图节点维护的一个参数，在计算梯度时可以用于在 `forward` 和 `backward` 方法间传递信息。

**在 PyTorch 的自定义 `Function` 中，`ctx` 必须作为静态方法的第一个参数**，这是 PyTorch 框架的硬性约定。

在 `backward` 方法中最常用的就是 `ctx.save_for_backward()` 方法，它可以保存张量，供反向传播使用

```python
class CustomFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        # 保存输入张量，供backward使用
        ctx.save_for_backward(input, weight)
        
        # 执行前向计算
        output = input * weight
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的张量
        input, weight = ctx.saved_tensors
        
        # 计算梯度（偏导）
        grad_input = grad_output * weight
        grad_weight = grad_output * input
        
        return grad_input, grad_weight
```



`ctx.mark_non_differentiable()` 方法可以标记某些输出为不可微分，但是只作为标记，需要开发者读取标记后进行特殊处理（即不求偏导）：

```python
class ArgMaxFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # argmax的结果是索引，不可微分
        result = torch.argmax(input, dim=1)
        ctx.mark_non_differentiable(result)
        ctx.save_for_backward(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # 对于不可微分的输出，grad_output为None
        input, = ctx.saved_tensors
        # 返回与输入对应的梯度（这里简化处理）
        return torch.zeros_like(input)
```



从 `backward` 方法的说明中我们也能看出，**自动微分的本质就是局部梯度的组合**：

- 每个操作都是一个"黑盒"，只需要知道输入到输出的关系
- 反向传播时，每个黑盒只需要知道：给定输出梯度，如何计算输入梯度
- 整个系统的梯度就是这些局部梯度通过链式法则组合起来的结果





## 4 正则化

本部分参考了https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html

第一章的引入部分我们介绍了过拟合，而用于对抗过拟合的技术就被称为**正则化**（regularization）。 

实际上不是所有能缓解过拟合的操作都能被叫做正则化，最经典的例子就是使用更大更全面的训练集，可以让模型更好的学习真实模式，从而得出更准确的决策边界，但是这并不是正则化方式

正则化方式的另一个核心目标是**控制模型复杂度、提升泛化能力**。

- 优化效率：改善优化过程带来的另一个好处是允许我们使用一个物理上参数更多、潜力更大的模型，但通过正则化来限制，或者说优化它的时间复杂度



### 4.1 训练误差与泛化误差

为了进一步讨论这一现象，我们需要了解训练误差和泛化误差。 **训练误差**（training error）是指， 模型在训练数据集上计算得到的误差。 **泛化误差**（generalization error）是指， 模型应用在同样从原始样本的分布中抽取的无限多数据样本时，模型误差的期望。问题是，**泛化误差永远不可能被准确计算出来， 这是因为无限多的数据样本是一个虚构的对象。** 在实际中，我们只能通过将模型应用于一个独立的测试集来估计泛化误差， 该测试集由随机选取的、未曾在训练集中出现的数据样本构成。

下面的思维实验将有助于更好地说明这种情况。 假设一个大学生正在努力准备期末考试。 一个勤奋的学生会努力做好练习，并利用往年的考试题目来测试自己的能力。 尽管如此，在过去的考试题目上取得好成绩并不能保证他会在真正考试时发挥出色。 例如，学生可能试图通过死记硬背考题的答案来做准备。 他甚至可以完全记住过去考试的答案。 另一名学生可能会通过试图理解给出某些答案的原因来做准备。 在大多数情况下，后者会考得更好。



### 4.2 模型选择

在机器学习中，我们通常在评估几个候选模型后选择最终的模型。 这个过程叫做**模型选择**。 有时，需要进行比较的模型在本质上是完全不同的（比如，决策树与线性模型）。 又有时，我们需要比较不同的超参数设置下的同一类模型。

例如，训练多层感知机模型时，我们可能希望比较具有 不同数量的隐藏层、不同数量的隐藏单元以及不同的激活函数组合的模型。 为了确定候选模型中的最佳模型，我们通常会使用验证集。

#### 4.2.1 模型复杂性

当模型简单且数据量充足时，泛化误差通常会接近训练误差；相反，如果模型更复杂而样本数量较少，训练误差可能会降低，但泛化误差反而可能上升。

那么，什么是模型的复杂性？这个问题并不简单。一个模型的泛化能力受多种因素影响：例如，参数越多的模型通常越复杂；参数取值范围越大，也可能增加复杂性。在神经网络中，通常需要更多训练轮次的模型较为复杂，而只需较少轮次、甚至需要“早停”的模型则相对简单。

不过，要比较不同类型模型（如决策树与神经网络）之间的复杂性并不容易。目前，我们可以参考一条实用的经验法则：**一个模型如果能够轻易解释任何现象，那它很可能过于复杂；而一个表达能力有限、却仍能较好拟合数据的模型，往往更具实用价值**。这一观点在哲学上与波普尔提出的“可证伪性”标准相呼应：一个好的理论应当能够拟合观测数据，同时也必须能够被实际检验所推翻。这一点之所以重要，是因为所有统计估计都是“事后归纳”——即在观察到数据之后进行的推断，因此容易受到各种关联谬误的影响。

关联谬误在深度学习领域可以概括为：**对非本质模式的错误识别和依赖，导致在面对新情况时失去效能。**

>关联谬误，也称为“罪恶关联谬误”或“负面关联谬误”，是一种逻辑谬误。**它的核心在于错误地将经验跃升于逻辑之上**，例如将一个观点、理论或建议，与一个被普遍厌恶、恐惧或鄙视的人、群体、事物或观念联系起来，从而试图让人们不假思索地拒绝该观点。

当然，我们在此不必过多探讨哲学问题，还是回到更实际的讨论上来。

为了建立直观理解，这里概括性的给出几个通常会影响模型泛化能力的关键因素：

- **可学习参数的数量**：参数数量（也常被称为“自由度”）越多，模型往往越容易过拟合。
- **参数的取值大小**：当模型中的权重取值范围较大时，也更容易出现过拟合现象。
- **训练样本的数量**：训练数据的规模同样至关重要。即便是简单模型，也容易过拟合仅含一两个样本的数据集；而要过拟合一个包含数百万样本的数据集，则需要一个极为复杂的模型。

对于可学习的参数数量，我们这里用多项式函数作为对决策边界的拟合，高阶多项式函数通常比低阶多项式复杂得多。一方面，它们参数更多，另一方面，它们所能表示的函数范围也更广。因此，在同一个训练数据集上，高阶多项式的训练误差理论上不会大于低阶多项式——最坏情况下两者相当，多数时候会更低。

事实上，只要样本点 x 的值互不相同，我们甚至可以使用一个阶数等于样本数量的多项式来完美地拟合整个训练集，达到训练误差为零，但是过度拟合训练集也会导致模型更大概率学习到上述提到的“关联谬误”，而不是真实模式。

下图展示了欠拟合、过拟合及它们与训练误差与泛化误差的关系：

![../_images/capacity-vs-error.svg](assets\capacity-vs-error.png)



### 4.3 验证集

在模型的所有超参数确定之前，我们原则上不应使用测试集。如果在模型选择过程中依赖测试数据，可能会导致模型对测试集产生过拟合，这将带来严重问题。当我们对训练集过拟合时，还可以通过测试集上的表现来判断；但若过拟合的是测试集本身，我们就失去了评估泛化能力的可靠依据。因此，绝不能依赖测试集进行模型选择，但反过来，如果只依赖训练数据，我们也无法准确估计模型的真实泛化误差。

实际应用中情况更为复杂。理想情况下，测试集应仅使用一次——用于最终评估或模型比较。但现实中，我们很少能在每次实验时都使用全新的测试集，测试数据往往会被反复使用。

常见的解决方案是将数据划分为三部分：**除了训练集和测试集之外，增加一个“验证集”**。但值得注意的是，实践中验证集与测试集之间的界限常常模糊不清。一种常见的实践是将训练集的一部分划作验证集，**验证集不参与训练，只作为训练过程中可以反复使用的评估指标。**



## 5 定义网络

`nn.Module` 是 PyTorch 中 `torch.nn` 模块提供的一个基类，所有神经网络模块（包括层和模型）都继承自它。用户可以通过继承 `nn.Module` 来定义自己的神经网络模型。

在 PyTorch 中，当我们通过继承 `torch.nn.Module` 类来定义自己的模型时，通常需要实现两个关键方法：

1. **`__init__` 方法（初始化）**：
   - 用于定义模型中使用的**层（layers）**、**参数（parameters）** 和**子模块（submodules）**。
   - 例如：线性层（`nn.Linear`）、卷积层（`nn.Conv2d`）、激活函数（如 `nn.ReLU`）、Dropout 等。
   - 这些组件在 `__init__` 中被创建并赋值为类的属性（通常以 `self.xxx` 的形式）。
2. **`forward` 方法（前向传播 / 数据流向定义）**：
   - 定义输入数据如何通过模型中的各个层进行计算，最终得到输出。
   - PyTorch 会自动根据 `forward` 中的操作构建计算图，用于反向传播和梯度计算。
   - 用户**不需要**显式定义 `backward` 方法（除非有特殊需求），PyTorch 会自动处理。

**通常情况下不需要在定义网络时关注反向传播 `backward` 过程**，因为在层次化的人工神经网络中，如果需要自定义反向传播，**应该在对应的操作/层级别实现**，而不是在网络整体层面处理。

但是 `forward` 方法对模型的运行很重要，因为 pytorch 框架是依据该方法对网络的组织构建运算图，并自动求偏导、反向传播的。也即：**在 PyTorch 中，`__init__` 定义了网络的组成部件，而 `forward()` 的执行过程才真正构建计算图并决定网络的运算路径。**

此外，对于绝大多数在pytorch框架内或者数据流最终流回pytorch计算图的 `nn.Module` ，这些模块有着**调用上的一致性**：它们都可以用 `module(input)` 的方式调用

### 5.1 直接继承

通过上面的说明，我们可以用如下方法定义一个三层的人工神经网络：

```python
#导入所需的库
import torch
import torch.nn as nn
import torch.nn.functional as F

#继承 nn.Module 并重写 __init__ 和 forward 方法。
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(784, 256)   # 输入层 -> 隐藏层1
        self.fc2 = nn.Linear(256, 128)   # 隐藏层1 -> 隐藏层2
        self.fc3 = nn.Linear(128, 10)    # 隐藏层2 -> 输出层

    def forward(self, x):
        # 定义前向传播过程
        x = F.relu(self.fc1(x))   # 第一层 + ReLU 激活
        x = F.relu(self.fc2(x))   # 第二层 + ReLU 激活
        x = self.fc3(x)           # 输出层（未加 softmax）
        return x

#创建模型实例
model = MyNet()
print(model)

```

如上所述，`forward()` 方法才是真正构建网络层次。

上述代码的输出为

```
MyNet(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
)
```



因为 pytorch 将网络的输出和激活函数分开处理，所以对于形式固定的激活函数，通常都是在 `forward` 方法中直接使用开箱即用的 pytorch 类的。

但是，对于带可学习参数的激活函数，如 `PReLU` ，因为**其作为网络训练的一部分，而不止是对输出进行固定的非线性变换**，所以必须在 `__init__()` 方法中进行定义

示例代码如下

```python
import torch
import torch.nn as nn

class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # 定义激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))   # 使用在 __init__ 中定义的 ReLU
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)          # 输出用 Softmax
        return x

#创建模型实例
model = MyNet2()
print(model)
```

上述代码输出为：

```python
MyNet2(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
  (relu): ReLU()
  (softmax): Softmax(dim=1)
)
```



对于已有模型的调试以及二次开发时通常需要对模型的中间输出进行提取，pytorch也提供了对每层进行命名的功能，**在继承 `torch.Module` 定义的网络中，每层的变量名就是该层的名字**

### 5.2 使用 Sequential

`nn.Sequential` 是对 `nn.Module` 的封装，它提供了一种一种简单而强大的方式来定义神经网络，特别适用于**层按顺序堆叠**的网络结构。

`Sequential` 容器会按照添加的顺序依次执行其中的模块，前一层的输出自动作为后一层的输入。

简单示例如下：

```python
# 定义一个简单的全连接网络
model = nn.Sequential(
    nn.Linear(784, 128),    # 输入784维，输出128维
    nn.ReLU(),              # ReLU激活函数
    nn.Linear(128, 64),     # 输入128维，输出64维
    nn.ReLU(),              # ReLU激活函数
    nn.Linear(64, 10)       # 输出10类（如MNIST分类）
)
print(model)
```

上述测试代码的输出为：

```
Sequential(
  (0): Linear(in_features=784, out_features=128, bias=True)
  (1): ReLU()
  (2): Linear(in_features=128, out_features=64, bias=True) 
  (3): ReLU()
  (4): Linear(in_features=64, out_features=10, bias=True)  
)
```



同样的，对于使用 Sequential 定义的网络，也可以方便的定义每层的名字，对于复杂的命名需求，也可以用 OrderedDict 来实现

```python
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc_layer_1', nn.Linear(784, 128)),
    ('relu_layer_1', nn.ReLU()),
    ('fc_layer_2', nn.Linear(128, 64)),
    ('relu_layer_2', nn.ReLU()),
    ('output', nn.Linear(64, 10))
]))
print(model)
```

上述测试代码的输出为：

```
Sequential(
  (fc_layer_1): Linear(in_features=784, out_features=128, bias=True)
  (relu_layer_1): ReLU()
  (fc_layer_2): Linear(in_features=128, out_features=64, bias=True)
  (relu_layer_2): ReLU()
  (output): Linear(in_features=64, out_features=10, bias=True)
)
```



## 6 训练

对于人工神经网络的训练，可以概括为以下几步

1. **加载数据**（数据预处理、划分训练/验证集）
2. **定义网络结构**
3. **定义损失函数和优化器**
4. **训练循环**（重复以下步骤直到达到预定epoch）：
   - 4.1 **获取批次数据**（从数据加载器中取出一个batch）
   - 4.2 **前馈计算**（输入数据通过网络，得到预测输出）
   - 4.3 **计算损失**（比较预测值与真实值）
   - 4.4 **梯度清零**（将优化器的梯度缓冲区清零）
   - 4.5 **反向传播**（计算损失对各参数的梯度）
   - 4.6 **参数更新**（优化器根据梯度更新网络权重）

测试代码：

```python
# mnist_simple_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# 1. 加载数据（含预处理和划分）
# ----------------------------
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
    ])
    
    # 训练集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 验证集（从训练集中划分，或使用原始测试集作为验证）
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ----------------------------
# 2. 定义网络结构
# ----------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ----------------------------
# 3. 定义损失函数和优化器
# ----------------------------
def get_criterion_and_optimizer(model, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return criterion, optimizer

# ----------------------------
# 4. 训练循环
# ----------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # 4.1 获取批次数据（已由 DataLoader 提供）
        # 4.2 前馈计算
        output = model(data)
        # 4.3 计算损失
        loss = criterion(output, target)
        # 4.4 梯度清零
        optimizer.zero_grad()
        # 4.5 反向传播
        loss.backward()
        # 4.6 参数更新
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=False)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ----------------------------
# 验证函数（用于评估模型）
# ----------------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ----------------------------
# 主函数
# ----------------------------
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 超参数
    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    
    # 1. 加载数据
    train_loader, val_loader = load_data(batch_size=batch_size)
    
    # 2. 定义网络
    model = SimpleNN().to(device)
    
    # 3. 定义损失函数和优化器
    criterion, optimizer = get_criterion_and_optimizer(model, lr=learning_rate)
    
    # 4. 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    main()
```



## 7 模型的保存和加载

当保存和加载模型时，需要熟悉三个核心功能：

1. `torch.save`：将序列化对象保存到磁盘。此函数使用Python的`pickle`模块进行序列化。使用此函数可以保存如模型、tensor、字典等各种对象。
2. `torch.load`：使用pickle的`unpickling`功能将pickle对象文件反序列化到内存。此功能还可以有助于设备加载数据。
3. `torch.nn.Module.load_state_dict`：使用反序列化函数 state_dict 来加载模型的参数字典。



### 7.1 状态字典 state_dict

在优化器、学习率调度器中我们使用过 `state_dict` 来保存对象的关键信息，同样的，模型的很多信息也被保存在 `state_dict` 中：

1. **模型参数（Parameters）**

这些是通过 `nn.Parameter` 注册的可学习参数，通常是权重（weights）和偏置（biases）。例如：

- `layer.weight`
- `layer.bias`

这些参数在优化过程中会被优化器更新。

2. **缓冲区（Buffers）**

缓冲区是模型中**不需要梯度**、但需要随模型一起保存的状态变量。它们通过 `register_buffer()` 方法注册。常见的例子包括：

- Batch Normalization 层中的 `running_mean` 和 `running_var`
- 某些自定义层中的统计量或固定状态

缓冲区不会被优化器更新，但对模型前向传播有影响，因此必须保存。



但是需要注意，**模型的 `state_dict` 不包括以下内容**：

- **模型结构（architecture）**：`state_dict` 只包含参数值，不包含网络结构。加载时需要先定义相同的模型结构。
- **优化器状态**：如动量、Adam 的 `m` 和 `v` 等，这些保存在 `optimizer.state_dict()` 中。



### 7.2 断点续训

断点续训（Checkpointing/Resume Training）是指在模型训练过程中保存当前状态，当训练意外中断（如服务器宕机、断电、手动停止等）后，能够从保存的检查点继续训练，而不是从头开始。

对于模型本身的断点保存方法已经有成熟的应用了，在下面会介绍。

但是对于数据集相关类型，或者说 **Dataset**, **Sampler** 和 **DataLoader** 这三个类需要保存的信息则没有通用的方法，因为通常情况下数据的提取只和训练的轮次数相关，只需要知道训练到哪个batch就行（通常情况下也不会要求断点能精确到批次内的某个输入），但是某些随机或者带状态的 Sampler 则需要单独处理。

> 对于断点续训，在学术场景下只需要保存特定epoch后的模型即可，因此数据提取相关类型不需要进行断点保存

通常情况下，断点需要保存如下信息：

1. 模型状态

   - `model.state_dict()`
   - 含义：保存所有可学习参数（权重、偏置等）。
   - 用途：恢复模型到中断时的参数状态。
   - 注意：只保存 `state_dict()`，不直接保存整个模型对象以节约存储空间

2. 优化器状态

   - `optimizer.state_dict()`
   - 含义：保存优化器内部的动量、平方梯度累积等信息。
   - 用途：继续训练时保持优化过程连续（例如 Adam 的动量不能丢）。
   - 举例：Adam/SGD 内部都有缓存梯度动量、学习率调度参数等。

3. 学习率调度器状态

   - `scheduler.state_dict()`
   - 含义：记录当前学习率、衰减阶段等。
   - 用途：恢复学习率变化进度，防止“断点重启后学习率突变”。

4. 当前训练进度信息

   - ```python
     {
         "epoch": current_epoch,
         "global_step": global_step,  # 选用：如果每 batch 更新时记录
         "best_val_loss": best_val_loss,  # 选用：如有早停机制
     }
     
     ```

   - 用途：恢复训练循环的位置、进度，否则你重启时不知道从哪个 epoch 继续。

   - epoch影响训练轮次、学习率调度器等

5. 随机数状态：如果模型有依赖随机数的话需要保存，当然这个是可选的，因为随机数来源很多，而且学术上不要求模型训练必须原样复现

   - 用途：**确保数据增强、Dropout、BatchShuffle 等操作一致**。
   - 如果只追求继续训练（不完全复现），可不保存。

一个完整的模型断点保存示例代码如下：

```python
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, filepath)
```



模型的断点恢复代码为：

```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```



### 7.3 模型的保存和读取

对于学术使用，保存模型的 `state_dict` 并开源模型的定义代码就足以满足学术上的复现要求了。

而如果想保存整个模型，则可以用下面的代码

```python
# 保存整个模型
torch.save(model, 'model.pth')

# 加载整个模型
loaded_model = torch.load('model.pth')
```

但是这样做会导致：

- 文件体积较大
- 对模型类的定义有依赖

