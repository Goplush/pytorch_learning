[TOC]

https://docs.pytorch.org/docs/stable/search.html 这是Pytorch的文档搜索网址，可以方便的搜索到本文绝大部分方法和类型的文档

## 0 人工神经网络

在进入书中内容之前，有必要介绍一下人工神经网络这一基础概念，有助于我们理解本章第三节及以后的内容

本部分参考了以下网站

https://paddlepedia.readthedocs.io/

http://askabiologist.asu.edu/chinese-simplified/%E7%A5%9E%E7%BB%8F%E5%85%83%E7%9A%84%E8%A7%A3%E5%89%96%E5%AD%A6

https://www.ruanyifeng.com/blog/2017/07/neural-network.html

### 0.1 神经元&感知机

人工智能学科的目标一直是对人脑智慧的趋近与完善，既然思考的基础是神经元，如果能够"人造神经元"（artificial neuron），就能组成人工神经网络，模拟思考。

对于人的神经细胞，它通过自己的树突从它相邻的神经元那里接收信号。接着，这些信号被传递到细胞主体，也就是胞体。胞体对收到的信号进行处理、转发信：号离开胞体，沿着轴突一路向下到达突触。最后，信号离开突触，并被传递给下一个神经细胞。

而神经细胞的特点是可能哟一个或多个树突、但是只能有一个轴突，即**可能有多个输入，但是只能有一个输出**（但是这并不限制把一个输出输出给多个后续的神经元）。

根据上述结构，1957年 Frank Rosenblatt 提出了一种简单的人工神经元，被称之为感知机。早期的感知机结构和 MCP 模型相似，由一个输入层和一个输出层构成，因此也被称为“单层感知机”。感知机的输入层负责接收实数值的输入向量，输出层则为1或-1两个值。

![图1 感知机模型](assets\single_perceptron.png)



可以看出，神经元对于输出的是经过线性求和后将结果传入激活函数的，由于多个线性变化的叠加仍然是线性变化，所以为了能增加神经网络的拟合能力，通常采用非线性的激活函数，并直接将激活函数值作为输出（而对所有输入线性加权求和的特征被保留了下来）

### 0.2 神经网络与决策模型

对于人类来说，无论是组织语言，还是思考问题，其本质都是在基于已有的输入进行决策：在做数学题时，**我们根据已知信息从所有可能的方法中选择自认为最合适做题方法、在说话时我们根据语境（对话上下文、语气等信息）组织自己认为合适的语言。**

因此在数学上，我们可以对这种决策过程进行建模：

- 把已知的信息的“特征”抽象成向量，这些向量张成了输入空间$\mathbb{R}^n$
- 把所有可能输出抽象成一个**有限集**

决策的过程就是从输入空间映射到输出空间的一个函数，**而决策模型关注的，就是如何去拟合这些决策，或者说，就是在 $\mathbb{R}^n$ 中找到一条（或多条）阈值边界，把不同类别区分开。**

很显然，由于真实环境的复杂性，输入空间通常都是维度较高的超空间，而决策阈值边界也是这个超空间中的一个或多个超曲面。为了实现这种拟合，通常将神经元以层次化的网络形式组合起来，构成神经网络：

![img](assets\bg2017071205.png)

上图中，底层感知器接收外部输入，做出判断以后，再发出信号，作为上层感知器的输入，直至得到最后的输出。在现实情况下还可能有多个输出

> 图里，信号都是单向的，即下层感知器的输出总是上层感知器的输入。现实中，有可能发生循环传递，即 A 传给 B，B 传给 C，C 又传给 A，这称为"递归神经网络"（recurrent neural network），通常情况下不涉及。

但是只拿到输出还不够，为了实现决策，还需要为不同输出确定决策阈值，确定输出强度达到多少时做出何种决策

因此，机器学习要做的就是通过“学习”过程，拟合出合适的神经元对输入的权重和决策阈值——**这些会随着学习过程自动更新的参数，也被称为“可学习” (learnable)参数**



### 0.3 损失函数

在初始化后，神经网络总是会给出一些初始输出（决策），神经网络学习的过程，就是要让输出决策不断地向真实决策靠拢，因此需要一个衡量模型输出与真实情况差异的函数，这就是损失函数

形式化地：

设：
- $ y \in \mathcal{Y} $ 为真实标签（ground truth），
- $ \hat{y} = f(x; \theta) \in \mathcal{Y} $ 为模型对输入 $ x $ 的预测值，
- $ \theta $ 为模型参数，

则损失函数是一个映射：
$$
\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}
$$

即：
$$
\ell(y, \hat{y}) = \ell(y, f(x; \theta))
$$

损失函数的值越大，表示预测越差；值越小，表示预测越准确。

损失函数的核心目标是为优化算法提供一个可微（或至少可优化）的目标函数，通过调整参数 $ \theta $，使总损失最小化：

$$
\min_{\theta} \sum_{i=1}^{N} \ell(y_i, f(x_i; \theta)) \quad \text{或} \quad \min_{\theta} \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(y, f(x; \theta))]
$$

其中第一个是经验风险最小化（Empirical Risk Minimization, ERM），基于训练集；

- 其中的$ x_i $与$y_i$分别指单个输入向量与其对应的真实输出
- $ \{(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)\} $ 是训练数据集，是来自真实分布 $ \mathcal{D} $ 的有限个观测样本。

第二个是期望风险最小化，基于数据分布 $ \mathcal{D} $，是理论目标。

- $ (x, y) \sim \mathcal{D} $ 表示：输入-标签对 $ (x, y) $ 是从某个未知的真实联合概率分布 $ \mathcal{D} $ 中独立采样得到的随机变量。
- 因此，$ x $ 是一个随机向量（random vector）是“任意一个可能的输入，按分布 $ \mathcal{D} $ 出现”。同理，$ y $ 是一个随机变量（或随机向量，在多输出情况下）。
- $ \mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(y, f(x; \theta))] $ 是一个关于参数 $ \theta $ 的函数，表示：在真实数据分布下，模型预测的平均损失。
- 我们希望找到最优参数 $ \theta^* $，使得这个“理论上的平均误差”最小。

如果分布 $ \mathcal{D} $ 有概率密度函数 $ p(x, y) $，则：

$$
\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(y, f(x; \theta))] = \iint \ell(y, f(x; \theta)) \, p(x, y) \, dx \, dy
$$

如果是离散分布，则是求和：

$$
\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} \ell(y, f(x; \theta)) \cdot P(X = x, Y = y)
$$
最后说一下损失函数的一些常见要求

1. **非负性**

   $$
   \ell(y, \hat{y}) \geq 0, \quad \forall y, \hat{y}
   $$

   通常当且仅当 $ y = \hat{y} $ 时取 0（但非必须）。

2. **连续性与可微性**：可微性对梯度下降至关重要。

   

3. **凸性（Convexity）**：若损失函数关于预测值 $ \hat{y} $ 是凸函数，则优化问题更容易（局部最优 = 全局最优）。

   - L2、Hinge、Logistic Loss 是凸的；

   - 神经网络中的损失函数因 $ f(x; \theta) $ 非线性，整体非凸。

4. 可解释性：有些损失函数具有良好的可解释性，可以提供有关模型性能的直观理解。例如，对于分类问题，交叉熵损失函数可以解释为最小化模型对真实类别的不确定性。

5. ……



### 0.4 梯度下降

在通过损失函数衡量预测决策与真实决策的差异，来对模型进行自动优化时，**通常采用沿损失函数梯度下降，并反向传播，更新权重的方法进行。**因此激活函数和损失函数是需要对参数求梯度的

> 可以看作是从损失函数开始逐层对可学习参数求偏导

由于层次化的神经网络可以看作是多层复合函数，求梯度时就会涉及到梯度相乘，加上同一个网络的神经元通常采用一致的激活函数，因此梯度在某些情况下会指数级变化，产生以下两个问题

- 梯度消失（Vanishing Gradient）：在反向传播过程中，**梯度值随着层数加深而指数级衰减**，导致浅层（靠近输入层）参数几乎得不到有效更新，模型训练停滞。
- 梯度爆炸（Exploding Gradient）：在反向传播过程中，**梯度值随着层数加深而指数级增长**，导致参数更新幅度过大，损失函数震荡甚至发散（NaN）

一般情况下会**通过对权重初始值的设置、网路结构的设置、激活函数的选择等多方面来避免梯度消失和梯度爆炸的问题**

## 1 安装 pytorch

### 1.1 CUDA

CUDA（统一计算设备架构）是由 NVIDIA 开发的并行计算平台和编程模型，允许开发人员利用 NVIDIA GPU 的强大功能执行通用的计算密集型任务，从而显著加速深度学习、科学模拟和图像处理等领域的应用程序

它的工作原理是利用 GPU 中数千个并行处理核心同时运行大量线程，这对于可并行化任务而言比 CPU 快得多。

虽然 pytorch 并不必须依赖CUDA，但是通过 CPU 运行pytorch框架效率很低，工程上不支持训练或运行大模型。因此，如果使用Nvidia显卡，最好使用依赖CUDA的pytorch。

PyTorch 已经**静态链接或打包了它所需的 CUDA 运行时库，所以只需要检查显卡支持的 CUDA 版本就行**

### 1.2 查看显卡支持的CUDA版本

**Windows**可以右键选择Nvidia控制面板，点击左下角的系统信息，再点击组件选项卡就可以显示本机显卡支持的CUDA版本了

![image-20250911175957144](assets\image-20250911175957144.png)

**通用的**命令行方法是：在安装最新的Nvidia显卡驱动后在命令行中输入`nvidia-smi`就可以显示支持的CUDA版本了

![image-20250911180526289](assets\image-20250911180526289.png)



### 1.3 安装pytorch

在pytorch官网https://pytorch.org/get-started/locally/选择不高于显卡支持的cuda版本的pytorch，网站会自动给出安装命令

> 安装时不能挂VPN，会有SSL error

![image-20250911181826896](assets\image-20250911181826896.png)

如果命令行显示`Successfully installed ...`且没报错信息，就说明安装好了

可以在python文件开头通过`import torch`导入pytorch包

## 2 张量

tensor中的任意维度的向量

### 2.1 张量的数据类型

tensor可以存储以下几种数据类型，为了统一 CPU tensor 与 GPU tensor ，**为tensor对象设置了 dtype属性**来标志其中存储的数据类型

| 数据类型        | dtype                         | CPU tensor         | GPU tensor              |
| --------------- | ----------------------------- | ------------------ | ----------------------- |
| 32 位浮点型     | torch.float32 或 torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64 位浮点型     | torch.float64 或 torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16 位浮点型     | torch.float16 或 torch.half   | torch.HalfTensor   | torch.cuda.HalfTensor   |
| 8 位无符号整型  | torch.uint8                   | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8 位有符号整型  | torch.int8                    | torch.CharTensor   | torch.cuda.CharTensor   |
| 16 位有符号整型 | torch.int16 或 torch.short    | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32 位有符号整型 | torch.int32 或 torch.int      | torch.IntTensor    | torch.cuda.IntTensor    |
| 64 位有符号整型 | torch.int64 或 torch.long     | torch.LongTensor   | torch.cuda.LongTensor   |

> 表1 张量中可以存储的数据类型

可以直接输出dtype

```python
def dtype_test():
    #declare a tensor
    t_a=torch.tensor([1.2,3.4])
    #get it's type
    print("data type of tensor {t_a} is {dtype}".format(t_a=t_a,dtype=t_a.dtype))

#上述方法的输出为data type of tensor tensor([1.2000, 3.4000]) is torch.float32
```



对于张量的存储类型，可以通过`torch.set_default_dtype`方法指定tensor默认的存储类型（相同格式也有getter方法来获得默认存储类型），也可以用下面给出的数据转换方法进行数据转换

```python
def change_dtype():
    # 设置默认张量类型为 DoubleTensor
    torch.set_default_dtype(torch.DoubleTensor)
    print("默认张量类型:", torch.tensor([1.2, 3.4]).dtype)

    # 将张量进行数据类型转换
    a = torch.tensor([1.2, 3.4])
    print("a.dtype:", a.dtype)
    print("a.long() 方法:", a.long().dtype)
    print("a.int() 方法:", a.int().dtype)
    print("a.float() 方法:", a.float().dtype)
```

上述方法的输出为：

```
默认张量类型: torch.float64
a.dtype: torch.float64
a.long() 方法: torch.int64
a.int() 方法: torch.int32
a.float() 方法: torch.float32
```



### 2.2 张量的基本属性

pytorch有`torch.Size`用于类保存张量的尺寸，tensor对象的`shape`属性就是该类实例，保存该张量的尺寸信息

也可以通过tensor对象的`size()`方法得到`torch.Size`类保存的对象的尺寸信息

```python
def tensor_shape():
    tensor_4d = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]
            ],
            [
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]]
            ]
        ]
    )
    print("tensor实例对象的shape属性:",tensor_4d.shape)
    print("过tensor对象的size()方法:",tensor_4d.size())
```

上述方法的输出为

```
tensor实例对象的shape属性: torch.Size([2, 2, 2, 2])
tensor对象的size()方法: torch.Size([2, 2, 2, 2])
```

可以调用实例对象的`numel()`方法来得到张量中元素的数量

```python
def tensor_ele_num():
    tensor_4d = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]
            ],
            [
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]]
            ]
        ]
    )
    print("非静态的numel()方法:",tensor_4d.numel())
    
    #上述方法的输出为：非静态的numel()方法: 16
```



### 2.3 张量的生成

本部分的所有测试代码在`3_tensor_gen.py`中

#### 2.3.1 生成张量的一般方法

python的列表和序列可以通过`torch.tensor()`方法构建张量，该方法有如下几个重要的参数：

- `dtype`可以指定张量中数据的存储方式（取值见表1）
- `requires_grad`可以指定pytorch是否为该张量求偏导（求偏导的具体过程在 2.4 节中介绍）
  - 注意**只有浮点型张量允许计算梯度**

此外，`torch.Tensor(*size,...)`，即`Tensor`类的构造方法，也是构造张量的常用方法：它不仅可以通过列表、序列构建张量，还能根据形状参数构建特定形状的张量（直接输入形状即可，但是构造（匿名）`torch.Size`类型参数的可读性更高）

> 对于前面加星号`*`的参数，这是是 Python 中的“可变参数”语法，允许传入任意数量的位置参数，这些参数会被打包成一个元组，赋值给 参数变量，后文不再赘述

```python
def build_tensor():
    a=torch.tensor([[2,3],[4,5]],dtype=torch.float64)
    b=torch.Tensor([[2,3],[4,5]])
    c=torch.Tensor(2,3,3)
   #上一行等价于d=torch.Tensor(torch.Size([2,3,3]))

    print("torch.tensor()方法通过列表创建的张量:",a,"; 其数据类型为:",a.dtype)
    print("torch.Tensor()构造方法通过列表创建的张量:",b,"; 其不支持指定数据类型，因此构造出张量内存放的数据类型为",b.dtype)
    print("torch.Tensor()构造方法构造的指定形状为2*3*3的张量:",c)
```

上述方法的输出为：

```
torch.tensor()方法通过列表创建的张量: tensor([[2., 3.],
        [4., 5.]], dtype=torch.float64) ; 其数据类型为: torch.float64
torch.Tensor()构造方法通过列表创建的张量: tensor([[2., 3.],
        [4., 5.]]) ; 其不支持指定数据类型，因此构造出张量内存放的数据类型为 torch.float32
torch.Tensor()构造方法构造的指定形状为2*3*3的张量: tensor([[[6.0416e+20, 1.7334e-42, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00]],

        [[0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00]]])
```



**张量对象**还有`new_**(*size,dtype,...)`系列方法，同样是可以创建新张量

`size`参数最好是由多个代表不懂维度列数的数量逐个给出，但是也支持打包以`torch.Size`对象的形式传入

- `new_full(*size,fill_value,dtype,...)`
- `new_zeros(*size,dtype,...)`
- `new_ones(*size,dtype,...)`
- `new_empty(*size,dtype,...)`：创建一个特定大小的填充有未初始化数据的张量

```python
def new_xx_series():
    mat=torch.tensor([1.0, 2.0])
    shape=torch.Size([2,3,3])
    a=mat.new_full(size=shape,fill_value=1)
    b=mat.new_ones(size=shape)
    c=mat.new_zeros(size=shape)
    d=mat.new_empty(size=shape)

    
    print("用new_full方法生成的2*3*3全1张量:",a)
    print("用new_ones方法生成的2*3*3全1张量:",b)
    print("用new_zeros方法生成的2*3*3全0张量:",c)
    print("用new_empty方法生成的2*3*3未初始化张量:",a)
```



上述方法的输出为

```
用new_full方法生成的2*3*3全1张量: tensor([[[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]]])
用new_ones方法生成的2*3*3全1张量: tensor([[[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]]])
用new_zeros方法生成的2*3*3全0张量: tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]])
用new_empty方法生成的2*3*3未初始化张量: tensor([[[3.7473e+19, 8.2116e-43, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00]],

        [[0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00]]])
```

 

#### 2.3.2 生成和给定张量同形的张量

pytorch还提供了`torch.xx_like(a:torch.Tensor, dtype=a.dtype,...)`系列函数来创建和给定张量同形的张量，方法名中的`xx`规定了返回张量的特征，参数`dtype`规定了生成张量存储的数据类型。

- `torch.ones_like(torch.Tensor)`生成与参数张量同形的全1张量
- `torch.zeros_like(torch.Tensor)`生成与参数张量同形的全0张量
- `torch.rand_like(torch.Tensor)`生成与参数张量同形的随机张量，填充值在`[0,1)`之间随机选取

```python
def make_alike_tensor():
    # 创建示例张量
    original_tensor = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]
    )
    print("原始张量:\n", original_tensor)
    print("原始张量数据类型:", original_tensor.dtype)

    # 使用 torch.ones_like 生成同形全1张量
    ones_tensor = torch.ones_like(original_tensor)
    print("\ntorch.ones_like 生成的张量:\n", ones_tensor)
    print("数据类型是否一致:", ones_tensor.dtype == original_tensor.dtype)
    print("形状是否一致:", ones_tensor.shape == original_tensor.shape)

    # 使用 torch.zeros_like 生成同形全0张量
    zeros_tensor = torch.zeros_like(original_tensor)
    print("\ntorch.zeros_like 生成的张量:\n", zeros_tensor)

    # 使用 torch.rand_like 生成同形随机张量（值在 [0, 1) 之间）
    rand_tensor = torch.rand_like(original_tensor)
    print("\ntorch.rand_like 生成的随机张量:\n", rand_tensor)
    print("随机值范围检查（最小值 >= 0）:", rand_tensor.min().item() >= 0.0)
    print("随机值范围检查（最大值 < 1）:", rand_tensor.max().item() < 1.0)
```



上述代码执行结果为：

```
原始张量:
 tensor([[[1., 2.],
         [3., 4.]],

        [[5., 6.],
         [7., 8.]]])
原始张量数据类型: torch.float32

torch.ones_like 生成的张量:
 tensor([[[1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.]]])
数据类型是否一致: True
形状是否一致: True

torch.zeros_like 生成的张量:
 tensor([[[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]]])

torch.rand_like 生成的随机张量:
 tensor([[[0.2968, 0.7301],
         [0.8657, 0.7427]],

        [[0.8845, 0.2786],
         [0.0824, 0.3758]]])
随机值范围检查（最小值 >= 0）: True
随机值范围检查（最大值 < 1）: True
```



#### 2.3.3 生成伪随机张量

pytorch框架提供了丰富的伪随机数生成机制，这里给出一些基础方法：

在生成伪随机数之前，可以使用`manual_seed(seed)`方法，指定随机数的种子

`torch.normal(mean,std,...)`方法返回一系列服从正态分布的随机数组成的张量

- 参数`mean`和`std`都是张量，代表着每个输出元素要服从的正态分布的均值与标准差
- `mean`和`std`中元素有两种对应关系：一一对应与一对多
  - 二者一一对应时，不要求同形，只要求元素数目相同，返回的张量形状和`mean`相同
    - 例子：`torch.normal(mean=t_a,std=t_a)`
  - 一对多时，所有输出元素的均值均为`mean`，标准差按照`std`的顺序使用，返回张量的形状也和`std`相同
    - 例子：`torch.normal(mean=torch.tensor([1.]),std=tensor([1.2,3.4]))`
  - 测试代码中的例子更有代表性和说明性

`torch.randn(size,...)`和`torch.randn_like(input,...)`生成符合参数给定的形状，元素服从**标准正态分布**$N(0,1)$的张量

```python
def gen_normal_distribution_tensor():
    t_a=torch.tensor([1.2,3.4])
    tensor_4d = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]
            ],
            [
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]]
            ]
        ]
    )
    a=torch.normal(mean=t_a,std=t_a)
    b=torch.normal(mean=torch.tensor([1.]),std=tensor_4d)
    c=torch.randn(2,3)
    d=torch.randn_like(tensor_4d)

    print("torch.normal()方法生成的均值与张量\n",t_a,"\n和标准差与张量\n",t_a,"\n元素相同时生成的张量:\n",a)
    print("torch.normal()方法均值张量为标量",torch.tensor([1.]),"\n标准差张量\n",tensor_4d,"\n时生成的张量: \n",b)
    print("torch.randn() 方法生成的元素符合标准正态分布的2*3张量:\n",c)
    print("torch.randn_like()方法生成的和上述四维张量同形的，元素服从标准正态分布的张量:\n",d)
```

上述方法的执行结果为

```
torch.normal()方法生成的均值与张量
 tensor([1.2000, 3.4000]) 
和标准差与张量
 tensor([1.2000, 3.4000])
元素相同时生成的张量:
 tensor([2.3965, 7.3783])
torch.normal()方法均值张量为标量 tensor([1.])
标准差张量
 tensor([[[[ 1.,  2.],
          [ 3.,  4.]],

         [[ 5.,  6.],
          [ 7.,  8.]]],


        [[[ 9., 10.],
          [11., 12.]],

         [[13., 14.],
          [15., 16.]]]])
时生成的张量:
 tensor([[[[  0.2268,   0.3671],
          [  1.2672,   2.6697]],

         [[  5.7695,   4.5054],
          [  2.2616,  16.4791]]],


        [[[  4.6464,  13.7602],
          [ -5.1174,  -7.5472]],

         [[ 22.3513,  -2.4562],
          [-14.0318, -15.4964]]]])
torch.randn() 方法生成的元素符合标准正态分布的2*3张量:
 tensor([[-1.3399,  1.1266, -0.1710],
        [ 1.8124,  0.3606,  0.6289]])
torch.randn_like()方法生成的和上述四维张量同形的，元素服从标准正态分布的张量:
 tensor([[[[ 0.8083,  0.0964],
          [ 1.2379, -0.3048]],

         [[ 1.4839,  0.8873],
          [ 0.0815, -1.7230]]],


        [[[-0.3377, -0.0294],
          [ 0.0540,  0.1442]],

         [[-0.1713, -1.7854],
          [ 1.5875, -0.4259]]]])
```



此外，`torch.rand(*size,...)`与`torch.rand_like(input)`生成符合参数给定的形状，元素服从**[0,1)均匀分布**$U(0,1)$的张量

```python
def gen_uniform_tensor():
    template=torch.Tensor(2,3)
    a=torch.rand(2,3)
    b=torch.rand_like(template)

    print("torch.randn() 方法生成的元素服从[0,1)均匀分布的2*3张量:\n",a)
    print("torch.randn_like()方法生成的和\n",template,"\n同形的，元素服从[0,1)均匀分布的张量:\n",b)
```

上述方法的执行结果为

```
torch.randn() 方法生成的元素服从[0,1)均匀分布的2*3张量:
 tensor([[0.2495, 0.7794, 0.3637],
        [0.6874, 0.3767, 0.2902]])
torch.randn_like()方法生成的和
 tensor([[9.4777e-38, 1.9352e-42, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])
同形的，元素服从[0,1)均匀分布的张量:
 tensor([[0.8910, 0.4119, 0.8204],
        [0.4604, 0.8569, 0.5489]])
```



pytorch也提供了开箱即用的类似于洗牌算法的函数：

`torch.randperm(n)`将**[0, n)**间所有整数（不包括n）进行随机排序后以张量的形式输出

```
def shuffle_0_to_n(n:int):
    a=torch.randperm(n)
    
    print("利用torch.randperm将0 到 ",n,"-1 的所有整数打乱:",a)

#上述方法的输出为：利用torch.randperm将0 到  8 -1 的所有整数打乱: tensor([0, 6, 7, 2, 5, 4, 3, 1])
```



对于伪随机生成控制，pytorch框架还提供了专门的`torch.Generator`类，但是入门阶段不需要学到这么深

#### 2.3.4 获取特定间隔的张量

pytorch 提供了依照特定间隔生成张量的功能

`torch.arange(start=0, end, step=1,...)`函数返回元素从`start`开始，` end`（不包括）为边界，步长为`step`按顺序排列形成的张量

而`torch.linspace(start, end, steps,...)`方法则是返回从`start`开始，`end`结束的均匀选取的`steps`个元素顺序排列形成的张量，即从`start`开始，每次增加`(end-start)/(steps-1)`，直到达到`end`为止：
$$
(\text{start}, \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1}, \ldots, \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1}, \text{end})
$$
下面给出上面两个函数的两个差异

- `step` vs. `steps`
  - `torch.arange`方法参数中的`step`规定的是步长
  - `torch.linspace`方法参数中的`steps`规定的是总步数
- 二者返回的张量都是从`start`开始，但是结束点不一样
  - `torch.arange`方法以`step`步长为幅度增长，在小于`end`的最大处停止
  - `torch.linspace`方法严格在`end`处停止

此外，pytorch还提供**元素间以对数为间隔的张量**的生成函数

`torch.logspace(start, end, steps, base=10.0,...)`方法返回从$base^{start}$开始，$base^{end}$结束，每项的比值相同（**以均匀倍数增长**）的`steps`项顺序组成的张量：
$$
\left( \text{base}^{\text{start}}, \text{base}^{\left(\text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1}\right)}, \ldots, \text{base}^{\left(\text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1}\right)}, \text{base}^{\text{end}} \right)
$$
测试代码如下：

```python
def tensor_range_ops():
    print("1. torch.arange(0, 10, 2) —— 等差序列（步长控制）")
    arange_tensor = torch.arange(0, 10, 2)  # 从0开始，到10（不包含），步长2
    print("张量内容:", arange_tensor)
    print("说明：以 step=2 递增，严格小于 end=10，因此不包含10")


    print("2. torch.linspace(0, 10, 5) —— 等间距序列（步数控制）")
    linspace_tensor = torch.linspace(0, 10, 5)  # 从0到10（包含），均匀取5个点
    print("张量内容:", linspace_tensor)
    print("验证：间隔应为 (10-0)/(5-1) = 2.5 →", (linspace_tensor[1] - linspace_tensor[0]).item())


    print("3. torch.logspace(start, end, steps, base) —— 对数等比序列")
    logspace_tensor = torch.logspace(0, 2, 5, base=10.0)  # 10^0 到 10^2，取5个点
    print("张量内容:", logspace_tensor)
    print("验证：相邻元素比值应相同 →",
          (logspace_tensor[1] / logspace_tensor[0]).item(),
          "≈",
          (logspace_tensor[2] / logspace_tensor[1]).item())

```



上述方法的输出为：

```
1. torch.arange(0, 10, 2) —— 等差序列（步长控制）
张量内容: tensor([0, 2, 4, 6, 8])
说明：以 step=2 递增，严格小于 end=10，因此不包含10
2. torch.linspace(0, 10, 5) —— 等间距序列（步数控制）
张量内容: tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
验证：间隔应为 (10-0)/(5-1) = 2.5 → 2.5
3. torch.logspace(start, end, steps, base) —— 对数等比序列
张量内容: tensor([  1.0000,   3.1623,  10.0000,  31.6228, 100.0000])
验证：相邻元素比值应相同 → 3.1622776985168457 ≈ 3.1622776985168457
```



#### 2.3.5 直接生成特殊张量

上面各种生成方法中都涵盖了对于特殊张量（比如全零，全1，全部由特定元素组成，未初始化）的生成，对于这些特殊张量，pytorch还提供了**一系列静态的生成方法**

- `torch.zeros(*size,...)`：全 0 张量  
- `torch.ones(*size,...)`：全 1 张量  
- `torch.eye(dim,...)`：单位张量
- `torch.full(*size, fill_value,...)`：使用特定元素填充的具有特定形状的张量  
- `torch.empty(*size,...)`：只包含未初始化数据的空张量

上述给出的重要参数中，`size`可以包含多个数，代表生成向量的维度要求；`dim`为一个整数，因为单位向量通常都是方阵

```python
def static_special_tensor_gen():
    zeros_tensor = torch.zeros(2, 3)  # 2x3 的全0张量
    ones_tensor = torch.ones(2,3)
    id_tensor = torch.eye(3)  # 3x3 单位矩阵
    full_tensor = torch.full((2, 3), 3.14)  # 2x3，用 3.14 填充
    empty_tensor = torch.empty(2, 3)  # 2x3 未初始化张量

    print("torch.zeros()方法生成的2*3全零张量:\n",zeros_tensor)
    print("\ntorch.ones()方法生成的2*3全 1 张量:\n",ones_tensor)
    print("\ntorch.eye()方法生成的3阶单位向量:\n",id_tensor)
    print("\ntorch.full()方法生成的2*3全 1 张量:\n",full_tensor)
    print("\ntorch.empty()方法生成的2*3 空张量:\n**由于未初始化，每次生成的内容可能不一样**\n",empty_tensor)
```



上述方法的输出为：

```
torch.zeros()方法生成的2*3全零张量:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])

torch.ones()方法生成的2*3全 1 张量:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

torch.eye()方法生成的3阶单位向量:
 tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])

torch.full()方法生成的2*3张量,用3.14填充:
 tensor([[3.1400, 3.1400, 3.1400],
        [3.1400, 3.1400, 3.1400]])

torch.empty()方法生成的2*3 空张量:
**由于未初始化，每次生成的内容可能不一样**
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```



可以很明显的看出，本节介绍的静态方法和2.3.2节介绍的非静态`torch.xxx_like()`系列成员方法存在着一一对应的关系，下面给出二者的一些辨析

- 对于`torch.xxx_like`来说，`dtype`、`device`等参数默认是继承自主调张量的
- 对于静态方法来说，这些参数拥有固定的默认值





### 2.4 张量的基本操作

#### 2.4.1 对特定函数求梯度

类比高等数学中梯度的概念：设函数 $ f: \mathbb{R}^n \to \mathbb{R} $ 在点 $ \mathbf{x} = (x_1, x_2, \dots, x_n) $ 处可微，则其在该点的梯度gradient）定义为：
$$
\nabla f(\mathbf{x}) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)	
$$
pytorch中也可以对特定**张量函数**求梯度，而且过程和上述定义展现的“对每个变量分别求偏导”的思想是一致的，因此有如下两点

- 首先，从数学定义可以看出，被求梯度的函数应当是单实值函数，同理，在pytorch中，**被求梯度的张量函数的函数值也应当是零维张量（标量）**
  - 对于结果是多维张量的张量函数来说，可以可以通过将所有元素加权求和的方式将其转化为标量，权值张量一般也被称为`gradient`参数

- 其次，还是从数学定义出发，可以看出求梯度本身这个过程和张量这一数据结构没有关系，因此**对特定张量函数求梯度的实质是对该函数的每个自变量求偏导**

由此我们又能引出下面几个问题

- 对于构成张量函数的张量，pytorch框架**如何识别哪些张量中的元素是要求偏导的，哪些是不需要的？**
- pytorch框架是**如何为每个“自变量”元素求偏导**的？
- 对一个张量函数，对一个自变量求完偏导后，这个**偏导数值应当存放在哪里？**



首先，**对于“哪些元素要求偏导”的问题**，在 2.3.1 节已经给出了答案——`requires_grad`可以指定pytorch是否为该张量求偏导

再说**“如何求偏导”**：同样是类比数学方法——链式求导：
$$
\text{对 } z = f(x, y)\text{，} \quad x = x(u, v)\text{，} \quad y = y(u, v)\\

\begin{aligned}
\frac{\partial z}{\partial u} &= \frac{\partial f}{\partial x} \cdot \frac{\partial x}{\partial u} + \frac{\partial f}{\partial y} \cdot \frac{\partial y}{\partial u} \\

\end{aligned}
$$
pytorch也会根据函数的结构，维护一张**有向的“计算图”**，其起点节点为要求梯度的张量函数；中间节点为函数计算过程中产生的中间结果；终点节点为用户创建的，`requires_grad=True`的张量。求梯度时就依据计算图为每个叶子节点的每个元素求偏导。

**“偏导存在哪”**：对于设置了`requires_grad=True`的张量，pytorch框架会维护**张量对象的`grad`属性**，这个属性是一个和对象同形的张量，其中每个元素存放的就是对应位置元素的偏导数值

最后，**求梯度的方法**就很简单了，只需要**调用目标张量函数的`backward()`方法**即可

示例代码如下，其中计算图的叶子节点是`W=[1.0, 2.0]`与`b=[0.5]`，目标函数是`((Wx[3.0,4.0]^T)+b)^2`，可以用笔算一下梯度并和计算结果对比

```
def grad_test():

    # 定义两个可训练参数
    W = torch.tensor([1.0, 2.0], requires_grad=True)  # shape: [1, 2]
    b = torch.tensor([0.5], requires_grad=True)         # shape: [1]

    # 输入
    x = torch.tensor([3.0, 4.0])  # shape: [2]

    # 前向计算：线性模型 y = Wx + b
    y = torch.matmul(W, x) + b  # 计算结果为 1*3 + 2*4 + 0.5 = 11.5

    # 损失函数：MSE，假设目标是 0.0
    target = torch.tensor([0.0])
    loss = (y - target) ** 2  # 损失值为 (11.5-0)^2 = 132.25

    loss = loss.sum()  # 对损失求和（此处单元素张量求和不改变值）

    print("Before backward, loss =", loss)

    # 反向传播：计算梯度
    loss.backward()

    # 查看梯度
    print("After backward")
    print("W.grad =", W.grad)   # W的梯度：∂loss/∂W = 2*(y-target)*x = 2*11.5*[3,4] = [69, 92]
    print("b.grad =", b.grad)   # b的梯度：∂loss/∂b = 2*(y-target) = 2*11.5 = 23
```



上述方法的执行结果为

```
Before backward, loss = tensor(132.2500, grad_fn=<SumBackward0>)
After backward
W.grad = tensor([69., 92.])
b.grad = tensor([23.])
```



#### 2.4.2 和np数组互转

可以用`torch.as_tensor(nparr)`方法和`torch.from_numpy(nparr)`方法将np数组转化为张量

- `torch.as_tensor(nparr)`方法更灵活，若输入已是张量则直接返回
- 对于`torch.from_numpy(nparr)`方法，返回的张量与`nparr`**共享内存**。对张量的修改将反映在`ndarr`中，反之亦然。返回的张量不可调整大小。  

调用**张量对象的**`numpy()`方法即可返回与张量相同的numpy形式

```python
    def np2tensor2np():
        # 步骤1：使用 NumPy 生成一个 3x3 的全1数组
        F = np.ones((3, 3))
        print("NumPy 数组 F:")
        print(F)

        # 步骤2：使用 torch.as_tensor() 将 NumPy 数组转换为 PyTorch 张量
        Ftensor = torch.as_tensor(F)
        print("\n使用 torch.as_tensor() 转换后:")
        print(Ftensor)
        print("数据类型:", Ftensor.dtype)

        # 步骤3：使用 torch.from_numpy() 转换（注意：会共享内存）
        Ftensor2 = torch.from_numpy(F)
        print("\n使用 torch.from_numpy() 转换后:")
        print(Ftensor2)
        print("数据类型:", Ftensor2.dtype)

        # 步骤4：将 PyTorch 张量转换回 NumPy 数组
        Fnumpy = Ftensor.numpy()
        print("\n使用 .numpy() 转换回 NumPy 数组:")
        print(Fnumpy)

```



上述方法的执行结果如下，数据类型为双精度浮点是是因为numpy数组的默认数据类型是双精度浮点数

```
NumPy 数组 F:
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]

使用 torch.as_tensor() 转换后:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
数据类型: torch.float64

使用 torch.from_numpy() 转换后:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
数据类型: torch.float64

使用 .numpy() 转换回 NumPy 数组:
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```



#### 2.4.3 改变张量形状

在介绍“变形”这个操作之前，需要先明确**张量对象在内存中是如何保存的**：和普通的多维数组一样，**张量在内存中通常情况下和多维数组一样，是连续存放的**，但是对于较大的无法连续存放的张量，pytorch框架也能做到**一定程度上的分散存储**，并且不影响张量的计算。

可以用`torch.Tensor.data_ptr()`非静态成员方法得到张量对象的数据指针

而对于张量的形状，pytorch使用了view机制来管理张量的形状，用`shape`成员属性记录张量的形状，使相同顺序存储相同元素的张量可以共享同一数据空间，从而节约内存。

![img](assets\26Q9g.png)

在不改变张量元素数目的情况下，pytorch提供了两种相似的方法来改变张量对象的形状（**元素数量相同，形状不同**）：

- `torch.Tensor`对象可以调用非静态的`reshape(*shape)`方法，**返回变形后的张量**（官方文档中也把张量对象的`xx`方法记作`torch.Tensor.xx`方法，本文不采用，但是可以辅助搜索）
- `torch.reshape(input, shape)`同样，返回元素和排列顺序与`input`张量相同，形状由`shape`参数序列规定的张量

需要注意的是，**上面两种方法返回新的变形后的张量是有可能和原始张量共享内存的，因此不建议同时修改变形前后的张量**

- 具体来说是由于张量分片存储时分片方法仍然与视图有关，所以不能简单的共享内存，而是按照规则复制到新的内存空间中

张量对象的`view(*shape)`、`view_as(input)`这两个方法可以直接用给定的形状或者张量来修改张量的view并返回修改后的张量，但是**`view`系列方法只支持修改视图，如果遇到分片存储的张量会直接报错**

如果想要直接修改张量本身，可以调用张量对象的`resize_(*shape:int)`方法或者对象的`resize_as_(input:torch:Tensor)`方法

```python
def tensor_reshape():
    A=torch.Tensor(3,8)
    B=torch.Tensor(4,6)
    a=A.reshape(2,12)
    b=A.reshape_as(B)
    A.resize_(2,12)
    B.resize_as_(A)

    print("初始张量:A=\n",A,"\nB=\n",B)
    print("\nA.reshape(2,12)返回的和a元素、排列相同，形状为2*12的张量a:\n",a)
    print("\n验证a 与 A 是否共享内存空间: ",a.data_ptr()==A.data_ptr())
    print("\nA.reshape_as(B)返回的和a元素、排列相同，形状同B的张量b:\n",b)
    print("\nA.resize_将A本身变成2*12：\n",A)
    print("\nB.resize_as_(A)将B本身变成2*12：\n",B)
```



上述方法的输出为：

```
初始张量:A=
 tensor([[3.5461e+13, 1.2065e-42, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
B=
 tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

A.reshape(2,12)返回的和a元素、排列相同，形状为2*12的张量a:
 tensor([[3.5461e+13, 1.2065e-42, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

验证a 与 A 是否共享内存空间:  True

A.reshape_as(B)返回的和a元素、排列相同，形状同B的张量b:
 tensor([[3.5461e+13, 1.2065e-42, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

A.resize_将A本身变成2*12：
 tensor([[3.5461e+13, 1.2065e-42, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

B.resize_as_(A)将B本身变成2*12：
 tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
```



此外，pytorch还支持在张量的特定维度前增加列数为1的新维度或者在去除特定的列数为1的维度，从而方便的改变张量的形状：

`torch.unsqueeze(input, dim)`对`input`张量在`dim`维度前添加新的维度

`torch.squeeze(input, dim)`去除`input`张量的`dim`指定的一个或多个列数为1的维度，请尽量保证指定去除的维度列数为1，否则目前属于UB

```python
def squeeze_tensor():
    x = torch.zeros(2, 1, 2, 1, 2)
    print("the shape of x is:",x.size())
    y=torch.squeeze(x,3)
    print("after squeeze the 3rd dim, the shape is:",y.size())
    y=torch.unsqueeze(y,3)
    print("unsing unsqueeze to put one col back to the 3rd dim, the shapeis:",x.size())
```



上述测试方法的输出为：

```
the shape of x is: torch.Size([2, 1, 2, 1, 2])
after squeeze the 3rd dim, the shape is: torch.Size([2, 1, 2, 2])
unsing unsqueeze to put one col back to the 3rd dim, the shapeis: torch.Size([2, 1, 2, 1, 2])
```



#### 2.4.4 获得张量的元素

张量和数组一样，支持通过索引`a[x][y]...`的形式访问

此外，张量还支持通过切片的形式实现对任意维度的元素定义独立的切片筛选规则后提取每一维度的索引都符合要求的元素构成返回张量，具体语法为：
$$
tensor[start_0:end_0:step_0,...,start_i:end_i:step_i,...]
$$
含义是：以`stepi`为步幅，`starti`为起点筛选第`i`维度下标在`[starti,endi)`范围内的元素。

简写规则与python本身一致，这里不赘述

```python
def tensor_slice():
    tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    print("原始张量为:\n",tensor)

    # 获取第0行
    slice1 = tensor[0]          # → [1, 2, 3]
    print("\n它的第0行为:",slice1)

    # 获取第1列
    slice2 = tensor[:, 1]       # → [2, 5, 8]
    print("\n它的第1列为:",slice2)

    # 获取前两行、前两列
    slice3 = tensor[:2, :1] 
    print("\n通过切片得到它的前两行，第一列元素为:\n",slice3)

    # 步长切片：每隔一行/列取一个
    slice4 = tensor[::2, ::2] 
    print("\n通过步长切片，每隔一行、一列取一个:\n",slice4)
```



上述方法的执行结果为：

```
原始张量为:
 tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

它的第0行为: tensor([1, 2, 3])

它的第1列为: tensor([2, 5, 8])

通过切片得到它的前两行，第一列元素为:
 tensor([[1],
        [4]])

通过步长切片，每隔一行、一列取一个:
 tensor([[1, 3],
        [7, 9]])
```





pytorch也支持返回**二维张量**的上/下三角矩阵形式

`torch.tril(input,diagonal=0)`返回下三角矩阵（`l`=low）

`torch.triu(input,diagonal=0)`返回上三角矩阵（`u`=up）

参数`diagonal`：

- 为0时返回三角矩阵的标准形式
- 为正值时在主对角线上方多保留/去除n条对角线
- 为负值时在主对角线下方多保留/去除n条对角线

对于非方阵，主对角线就是$[i,i]$

```
def tri_torch():
    a = torch.randn(4, 4)
    print("张量a:\n",a)
    b=torch.tril(a,0)
    print("\na的下三角矩阵形式:\n",b)
    c=torch.triu(a,1)
    print("\na的上三角矩阵形式，主对角线上方多去一条:\n",c)
```



上述方法的执行结果为：

```
张量a:
 tensor([[-0.4675, -1.5131,  0.1636,  0.5323],
        [-1.3956,  0.7944, -0.0284,  0.7135],
        [-0.2605, -0.4285,  0.7079,  0.6841],
        [ 0.2756, -0.8057, -1.5797,  0.5346]])

a的下三角矩阵形式:
 tensor([[-0.4675,  0.0000,  0.0000,  0.0000],
        [-1.3956,  0.7944,  0.0000,  0.0000],
        [-0.2605, -0.4285,  0.7079,  0.0000],
        [ 0.2756, -0.8057, -1.5797,  0.5346]])

a的上三角矩阵形式，主对角线上方多去一条:
 tensor([[ 0.0000, -1.5131,  0.1636,  0.5323],
        [ 0.0000,  0.0000, -0.0284,  0.7135],
        [ 0.0000,  0.0000,  0.0000,  0.6841],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])
```



#### 2.4.5 张量的拼接和拆分

`torch.cat(tensors,dim=0)`将`tensors`参数给出的一系列张量在`dim`规定的维度上**合并**。

`tensors`参数要求是一个张量组成的序列或元组

函数要求：

- 在张量包含的维度内合并（不支持自动扩充维度）
  - 即`dim`的取值范围不能超过张量自身的维度
- 除了要合并的维度外，张量在其他维度完全同形



`torch.stack(tensors,dim=0)`将`tensors`参数给出的一系列张量在`dim`规定的维度上**堆叠**。

`tensors`参数同样要求是一个张量组成的序列或元组

函数是**创建了一个新的维度来堆叠这些张量**，因此要求

- 要堆叠的张量完全同形
- `dim`最大取张量的维度+1



最后通过形状层面来辨析一下这两个方法

对于同形的`N`个给定的张量，他们的形状设为`(d1,d2,...,di)`

- 对于`torch.cat`方法，假设在第`k`维合并，那合并后张量的形状是：`d1,d2,...,dk*N,...,di`
- 对于`torch.stack`方法，假设在第`k`维堆叠，那堆叠之后张量的形状是：`d1,d2,...,d(k-1),N,dk,...,di`

```python
def joint_tensor():
    a=torch.arange(6.0).reshape(2,3)
    print("第一个张量a:\n",a)
    b=torch.randn_like(a)
    print("\n第二个张量b:\n",b)

    c=torch.cat((a,b),dim=1)
    print("\na,b在第1维度（列维度）拼接后的向量:\n",c)

    d=torch.stack((a,b),dim=2)
    print("\na,b在第2维度（新维度）拼接后的张量:\n",d)
```



上述代码的结果为：

```
第一个张量a:
 tensor([[0., 1., 2.],
        [3., 4., 5.]])

第二个张量b:
 tensor([[-0.4004,  0.7681,  0.1669],
        [ 0.5033,  0.1907,  1.5808]])

a,b在第1维度（列维度）拼接后的向量:
 tensor([[ 0.0000,  1.0000,  2.0000, -0.4004,  0.7681,  0.1669],
        [ 3.0000,  4.0000,  5.0000,  0.5033,  0.1907,  1.5808]])

a,b在第2维度（新维度）拼接后的张量:
 tensor([[[ 0.0000, -0.4004],
         [ 1.0000,  0.7681],
         [ 2.0000,  0.1669]],

        [[ 3.0000,  0.5033],
         [ 4.0000,  0.1907],
         [ 5.0000,  1.5808]]])
```



对于拆分张量，pytorch也提供了相关方法

`torch.chunk(input, chunks, dim = 0)`方法可以将`input`张量沿着`dim`维度拆分，总共拆成`chunk`块。如果不够分，会优先减少排序靠后的块的元素和维度，也可能返回小于`chunk`规定的块数

`torch.split(input, split_sizes, dim=0)`方法同样是将`input`张量沿`dim`维拆分，函数规定严格按照`split_sizes`拆分，该函数有以下两种模式

- `split_sizes`为整数时，试图拆分成这么多块，如果不够拆时优先减少靠后张量的维度，若目标`dim`小于`split_sizes`，属于UB情况，目前版本是返回原始张量
- `split_sizes`为列表时，试图拆分成`len(split_sizes)`块，每块包含的`dim`维列数由列表元素`split_sizes[i]`规定，如果不够拆，会报错

测试代码为：

```python
def split_tensor():
    a=torch.arange(27.0).reshape(3,3,3)
    print("原始张量a为:\n",a)

    b=torch.chunk(a,6,2)
    print(
        "\n想通过torch.chunk(a,6,2)把a沿着第[2]维拆分成六块，但是实际返回 ",
        b.__len__(),
        " 块，每块的内容为:\n"
    )
    for i in b:
        print(i,"\n")

    c=torch.split(a,2,0)
    print(
        "\n想通过torch.split(a,2,0)把a沿着第[0]维拆分成两块，实际返回 ",
        c.__len__(),
        " 块，每块的内容为:\n"
    )
    for i in c:
        print(i,"\n")
```



执行结果为：

```
原始张量a为:
 tensor([[[ 0.,  1.,  2.],
         [ 3.,  4.,  5.],
         [ 6.,  7.,  8.]],

        [[ 9., 10., 11.],
         [12., 13., 14.],
         [15., 16., 17.]],

        [[18., 19., 20.],
         [21., 22., 23.],
         [24., 25., 26.]]])

想通过torch.chunk(a,6,2)把a沿着第[2]维拆分成六块，但是实际返回  3  块，每块的内容为:

tensor([[[ 0.],
         [ 3.],
         [ 6.]],

        [[ 9.],
         [12.],
         [15.]],

        [[18.],
         [21.],
         [24.]]])

tensor([[[ 1.],
         [ 4.],
         [ 7.]],

        [[10.],
         [13.],
         [16.]],

        [[19.],
         [22.],
         [25.]]])

tensor([[[ 2.],
         [ 5.],
         [ 8.]],

        [[11.],
         [14.],
         [17.]],

        [[20.],
         [23.],
         [26.]]])


想通过torch.split(a,2,0)把a沿着第[0]维拆分成两块，实际返回  2  块，每块的内容为:

tensor([[[ 0.,  1.,  2.],
         [ 3.,  4.,  5.],
         [ 6.,  7.,  8.]],

        [[ 9., 10., 11.],
         [12., 13., 14.],
         [15., 16., 17.]]])

tensor([[[18., 19., 20.],
         [21., 22., 23.],
         [24., 25., 26.]]])
```



### 2.5 张量的计算

本部分很多函数理解起来都不难，再加上读者都有了一些pytorch基础了，所以就不会再给出很详细的示例代码了

#### 2.5.1 常用比较

| 函数                                                         | 功能                                                     |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| `torch.allclose(input, other, rtol = 1e-05, atol = 1e-08, equal_nan = False)` | 比较两个张量对应位置的元素是否接近，只有全接近才返回True |
| `torch.equal(input, other)`                                  | 判断两个张量是否具有相同的形状和元素                     |
| `torch.eq(input, other,...)`                                 | 逐元素比较是否相等                                       |
| `torch.ge(input, other,...)`                                 | 逐元素比较大于等于                                       |
| `torch.gt(input, other,...)`                                 | 逐元素比较大于                                           |
| `torch.le(input, other,...)`                                 | 逐元素比较小于等于                                       |
| `torch.lt(input, other,...)`                                 | 逐元素比较小于                                           |
| `torch.ne(input, other,...)`                                 | 逐元素比较不等于                                         |
| `torch.isnan()`                                              | 判断是否为缺失值                                         |

对于上述的**逐元素比较系列函数**

- 实际上pytorch引入了numpy中的**广播语义**来让不同形甚至元素数量不同的张量也可以进行逐元素操作，广播语义是pytorch中很重要的机制，建议学习，笔记在`extra/broadcast.md`中（如果想先过笔记的话，记住**最好只去比较两个同形张量**）。（[官方文档链接](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)）
- 它们返回一个和输入张量同形的布尔值张量

而对于`torch.allclose`方法，它的比较规则是
$$
\vert input_i-other_i\vert \le atol+rtol \times \vert other_i\vert
$$
注意它只返回一个布尔值，**说明两个张量每个对应位置的两个元素都在这种比较规则下是不是都足够接近**



#### 2.5.2 基本运算

pytorch支持对张量进行**逐元素**运算：加（`+`）、减（`-`）、乘（`*`）、除（`/`）、乘方（`**`）

`torch.sqrt(input,...)`是对输入张量进行开方运算

`torch.exp(input,...)`是对输入张量进行$e^x$运算

`torch.exp2(input,...)`是对输入张量进行$2^x$运算



除了逐元素的数值计算外，pytorch还对张量的矩阵运算提供了一定支持

`torch.matmul(input, other)`方法实现对两个任意维度的张量的乘法计算，该方法也可以用符号`@`代替

`torch.t(input)`方法返回转置后的张量，只应当传入二维张量

`torch.inverse(input)`方法返回张量的逆矩阵，只应当传入二维张量

`torch.trace(input)`方法返回张量的迹，即主对角线元素和



#### 2.5.3 统计相关的计算

pytorch提供了基础的数据剪裁方法

所谓剪裁，就是**通过将超过或低于一定阈值的元素值直接调整到阈值上**，来让数据维持在一定范围内的方法

`torch.clamp(input,min=None,max=None,...)`实现了数据剪裁

```
def clamp_test():
    a=torch.arange(0,8,1).reshape_as(torch.Tensor(2,4))
    b=a.clamp(min=3,max=5)

    print("初始张量为:\n",a,"\n经过[3,5]为阈值的剪裁后，变成:\n",b)

if __name__ == '__main__':
    clamp_test()
```



上述测试代码的输出为：

```
初始张量为:
 tensor([[0, 1, 2, 3],
        [4, 5, 6, 7]])
经过[3,5]为阈值的剪裁后，变成:
 tensor([[3, 3, 3, 3],
        [4, 5, 5, 5]])
```



下面给出一些基础的统计方法：

`torch.max`方法存在两个重载：

`torch.max(input)`仅返回输入张量的最大元素（以零维张量形式）

`torch.max(input, dim, keepdim=False,...)` 则返回两个张量`(values, indices)`，其中`values`为一个张量，每个元素的含义是其他维度取特定值时，张量沿`dim`维度能取到的最大值，第二个值为最大值所在的`dim`维度的下标组成的张量

这里具体解释一下”其他维度取特定值“的含义，假设`A`张量有`D`维，设形状为`a1*a2*...*aD`，那么`torch.max(input=A, dim=k, keepdim=False,...)`返回的`values`则有`D-1`维，形状为`a1*a2*...*a(k-1)*a(k+1)*....*aD`。相当于把`A`在`D`维超空间中沿着`k`维度投影，在其他坐标轴形成的超平面上的点，所以形状是上述的样子，而这个张量中每个元素的值就是`A`在取对应位置时**，`k`维方向上那一条线**（即该维度下的一条一维切片）上所有元素的最大值。三维情况下的示意图画在下面（沿x轴投影）。

![image-20250915205031062](assets\image-20250915205031062.png)

> 这一对于在`dim`维上进行xx操作的操作数选取方式比较常用，需要尽量理解

`keepdim`参数则决定要不要保留被投影（压缩）的`dim`维度，因为这个维度只有一列，可以被压缩而不影响数据的表示

`torch.argmax`方法同样有两个重载

`torch.argmax(input,...)` 输出`input`张量最大值元素所在的位置

`torch.argmax(input, dim, keepdim=False)`返回`torch.max(input, dim, keepdim=False,...)` 的第二个返回值

`torch.min(input,...)` 计算张量中的最小值，同理有两个重载，不赘述

`torch.argmin(input,...)` 输出最小值所在的位置，同样在不赘述





`torch.sort(input,dim=-1,descending=False,stable=False,...)`在`dim`维度上对`input`张量进行排序，每次单元排序的排序对象为：
$$
input[d_0,d_1,...,d_{dim-2},d_{dim},...]
$$
这一平行于`dim`维度的列向量，理解方式与上面`max`方法所述相同

`stable`参数表示排序是否需要为稳定排序，即大小相同的不同元素的相对位置是否可以改变





`torch.topk(input,k,dim=None,largest=True,sorted=True,...)`依据`largest`参数值返回`input`张量在`dim`维度的每一个列向量的前`k`个最大值或者最小值以及它们在该`dim`维列（该维度下的一维切片）中的位置（下标）

测试代码如下：

```python
def topk_test():
    a=torch.arange(0,27).reshape(3,3,3)
    print("初始张量a:\n",a)

    t2,t2p=torch.topk(a,2,2,True)

    print("\na沿最后一个维度的每一列的最大的两个值为:\n",t2,"\n它们在对应列（切片）中的位置分别是:\n",t2p)

```



运行输出如下：

```
初始张量a:
 tensor([[[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8]],

        [[ 9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]],

        [[18, 19, 20],
         [21, 22, 23],
         [24, 25, 26]]])

a沿最后一个维度的每一列的最大的两个值为:
 tensor([[[ 2,  1],
         [ 5,  4],
         [ 8,  7]],

        [[11, 10],
         [14, 13],
         [17, 16]],

        [[20, 19],
         [23, 22],
         [26, 25]]])
它们在对应列（切片）中的位置分别是:
 tensor([[[2, 1],
         [2, 1],
         [2, 1]],

        [[2, 1],
         [2, 1],
         [2, 1]],

        [[2, 1],
         [2, 1],
         [2, 1]]])
```



而`torch.kthvalue(input,k,dim=None,keepdim=False,...)`的行为则和`torch.max(input, dim, keepdim=False,...)`类似，是返回`dim`维第`k`小的数组成的张量

```python
def kthval_test():
    a=torch.randperm(27).reshape(3,3,3)
    print("张量a:\n",a)

    max,_=torch.max(a,2,False)
    print("\n用max得到a的第二维度最大值张量为:\n",max)

    min3,_=torch.kthvalue(a,3,2,False)
    print("\n用kthval得到a的第二维度最大值张量为:\n",min3)

    print("\n当参数合适时，max和kthvalue的行为是一致的:",torch.equal(min3,max))
```



上述方法的执行输出为：

```
张量a:
 tensor([[[13, 11, 14],
         [ 4, 19,  5],
         [26, 21, 23]],

        [[20, 18, 22],
         [ 2,  0,  1],
         [24,  7,  6]],

        [[16, 10,  3],
         [15, 12,  9],
         [25, 17,  8]]])

用max得到a的第二维度最大值张量为:
 tensor([[14, 19, 26],
        [22,  2, 24],
        [16, 15, 25]])

用kthval得到a的第二维度最大值张量为:
 tensor([[14, 19, 26],
        [22,  2, 24],
        [16, 15, 25]])

当参数合适时，max和kthvalue的行为是一致的: True
```



最后说一些统计量的计算函数

`torch.mean(input,dim=None,keepdim=False,...)`:  根据指定的维度计算张量的均值。  

- `input`: 输入的张量  
- `dim`: 指定要计算均值的维度（可以是整数或整数列表），若为 `None` 则对整个张量求均值。  
- `keepdim`: 布尔值，若为 `True`，则输出张量在指定维度上保留形状（即该维度大小为 1）；若为 `False`，则压缩该维度。  
- `dtype`: 输出张量的数据类型，若未指定，则使用输入张量的类型。

`torch.sum(input, dim=None, keepdim=False,...)`:  根据指定的维度对张量进行求和。  

- `input`: 输入的张量，元素需为数值类型。  
- `dim`: 指定求和的维度（可为整数或整数列表），若为 `None` 则对整个张量求和。  
- `keepdim`: 布尔值，决定是否保留求和维度的形状。  

`torch.median(input, dim=None, keepdim=False)`:  根据指定的维度计算张量的中位数。  

- `input`: 输入的张量，元素需为数值类型。  
- `dim`: 指定计算中位数的维度（可为整数或整数列表），若为 `None` 则对整个张量计算全局中位数。  
- `keepdim`: 布尔值，若为 `True`，则在输出中保留该维度的形状（大小为 1）；否则压缩该维度。  
- 返回值包含两个张量：第一个是中位数值，第二个是中位数对应的索引。

`torch.cumprod(input, dim, dtype=None)`:  根据指定的维度计算张量的累积乘积。  该函数沿指定维度逐个累乘元素，例如 `[1, 2, 3]` 在 `dim=0` 下结果为 `[1, 2, 6]`。

- `input`: 输入的张量，元素为数值类型。  
- `dim`: 指定进行累积乘积操作的维度（必须为整数）。  

`torch.std(input, unbiased=True, dim=None, keepdim=False, correction=0)`:  计算张量的标准差。  

- `input`: 输入的张量，元素为数值类型。  
- `unbiased`: 布尔值，若为 `True`，则使用 Bessel's correction（除以 $n-1$）计算无偏标准差；若为 `False`，则除以 $n$。  
- `dim`: 指定计算标准差的维度（可为整数或整数列表），若为 `None` 则对整个张量计算。  
- `keepdim`: 布尔值，决定是否保留计算维度的形状。  
- `correction`: 用于调整自由度的参数（默认为 0，等价于 `unbiased=False`；设为 1 时等价于 `unbiased=True`）。  
标准差是方差的平方根，反映数据离散程度。



## 3 torch.nn模块

`torch.nn` 是 PyTorch 中用于构建神经网络的核心模块，其设计目标是**简化神经网络的构建、训练和推理过程**，提供一套模块化、可复用、自动微分兼容的组件，让开发者能够快速搭建和训练深度学习模型。

### 3.1 卷积层

本部分的内容参考了

https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Convolution.html

#### 3.1.1 数学基础

这里需要说明以一个比较tricky的点：**在卷积神经网络中，卷积层的实现方式实际上是数学中定义的互相关 (cross-correlation)运算，而不是卷积(Convolution)运算**

在数学上，两个函数的 $ f(t) $ 和 $ g(t) $，它们的互相关定义为：
$$
(f \star g)(\tau) = \int_{-\infty}^{\infty} f(t)g(t + \tau)\,dt
$$
对于离散序列 $ f[n] $ 和 $ g[n] $​：
$$
(f \star g)[k] = \sum_{n=-\infty}^{\infty} f[n+k]g[n]
$$
其中$\tau$为偏移量

在pytorch中，上述的$g$就被称为卷积核，也被称为权重矩阵。卷积运算同样返回一个张量，其计算方式为就是离散卷积：
$$
out[i][j]=\sum_m\sum_nx[i+m,j+n]k[m,n]
$$
![图3 卷积计算过程](assets\convolution.png)

高维卷积同理，但是目前机器学习领域最常用的还是二维卷积（图像处理、计算机视觉）与三维卷积（视频分析（时空卷积）、医学图像（如CT/MRI体数据）、气象数据等）。更高维度的卷积可能出现在高维科学数据分析里（气象数据、流体仿真等），但实际应用场景较少。

卷积核在图像上滑动完毕后，其卷积结果形成的张量被称为特征图

#### 3.1.2 步长Stride

在卷积操作时，通常希望输出图像分辨率与输入图像分辨率相比会逐渐减少，即图像被约减。因此，可以通过改变卷积核在输入图像中移动步长大小来跳过一些像素，进行卷积滤波。当Stride=1时，卷积核滑动跳过1个像素，这是最基本的单步滑动，也是标准的卷积模式。Stride=k表示卷积核移动跳过的步长是k。下图展示了步长为2时的卷积过程

![图5 步幅为2的卷积过程](assets\stride.png)



#### 3.1.3 填充 padding

从上面的描述中我们也可以看出，输入图像边缘位置的像素点无法进行卷积，因此填充技术应运而生。填充是指在边缘像素点周围扩充元素，使得输入图像的边缘像素也可以参与卷积计算。注意，**在填充机制下，卷积后的图像分辨率将与卷积前图像分辨率一致，不存在下采样**。

pytorch的nn模块通过：

- `padding`控制填充大小

- `padding_mode`控制填充方式：

  - `zeros` 零填充：最常用，简单高效，但可能在边缘引入“黑边”或人为边界效应。

  - `reflect`反射填充（镜像）：以边界为镜面，反射内部数据进行填充，边缘过渡更自然，避免突变。

  - `replicate`复制填充：用最边缘的值向外复制填充，保持边缘值不变，避免引入新值

  - `circular`循环填充（环绕）：将数据视为循环结构，从另一端“绕回来”填充。适用于周期性数据（如角度、环形图像、信号）。



#### 3.1.4 扩张dilation

为了引入空洞卷积（dilated convolution），先介绍感受野这一概念

卷积所得结果中，每个特征图像素点取值依赖于输入图像中的某个区域，该区域被称为感受野（receptive field）。**在普通卷积的情况下，感受野内每个元素数值的变动，都会影响输出点的数值变化。**

增加感受野的一种只管方式就是增加卷积网络深度，即对输出的特征图再次进行卷积，如下图所示，通过两层3×3的卷积之后，最终特征图的感受野的大小将会增加到5×5，此时输出特征图中的一个像素点将会包含更多的图像语义信息。

![图7 感受野为5×5的卷积](assets\Receptive_Field_5x5.png)

但是增加卷积网络深度会增加整个网络的计算复杂度，在现实应用中算力也是需要考虑、节约的资源，因此提出空洞卷积的概念。

空洞卷积是将卷积核的采样点之间间隔特定距离（即跳过像素），使卷积核能“跨越”更大的空间范围，从而看到更广的上下文。例如在3*3的卷积核，扩张值`dilation=1`的情况下，卷积核的实际采样位置为：

```
X . X . X
. . . . .
X . X . X
. . . . .
X . X . X
```

**空洞卷积能增加输出特征图的感受野，但是也会造成特征图对微小变化敏感程度的下降**

> 空洞卷积只改变采样方式，步长不受扩张值的影响





#### 3.1.5 通道 channels 与 卷积核 filter

图像的尺寸只衡量了图像包含的像素个数，但是每个像素都可能包含更多特征（比如RGB分量的深度），这些特征在读取时就会被对应的**输入“通道**”读取、保存

同样的，我们**在对图像进行卷积时，可以针对不同的提取目标设计不同的卷积核**，每个卷积核会在输入的所有通道上滑动，并加权求和，生成一个**新的特征图**，**每个特征图就对应着一个输出通道，因此输出通道数 = 卷积核的数量**，而每个卷积核的形状就是

```
[in_channels, kernel_height, kernel_width]
```

![图9 多输出通道计算过程](assets\multi_out_channel.png)



在pytorch中，卷积核是用于提取图像特征的重要工具，它也是**可学习**的，因此pytorch把所有卷积核被按照输出通道组织在一起，存放在`torch.nn.Convxd.weight`变量中。其尺寸为：
$$
(out\_channels,\frac{in\_channels}{groups},kernel\_size[0],kernel\_size[1])
$$
其初始化值在$(-\sqrt k,\sqrt k)$中均匀选取，$k$为pytorch自动计算的数值

> 上面尺寸中的`groups`见下面的分组卷积

卷积核可学习**因为它是“如何提取图像特征”的核心 —— 它不是预先设定好的规则，而是通过数据训练出来的最优“滤波器”。**通过学习得到的滤波器是针对任务最有效的特征提取方式

对于输入通道，由于pytorch支持批处理，因此需要把图像转化为四维形式进行输出：

```
[batch,channel,height,width]
```

其中`batch`表示每批有多少张图片



#### 3.1.6 分组卷积

`torch.nn`模块还支持分组卷积，在分组卷积中，会把卷积核分成`groups`组，每一组的卷积核只在 `in_channels / groups` 个输入通道上滑动，并产生 `out_channels / groups` 个输出通道。 

> 意思是卷积核本身需要在输入通道维度上降维

最终所有输出通道**在通道维度上拼接（concatenate）**，得到最终完整的输出通道（和普通卷积数目相同）。

这种方式减弱了组间信息共性（不同组的输出通道来自于不同输入通道）和卷积核的信息组织能力（每个卷积都核不能统计所有输入通道的信息），换来了计算复杂度的下降

#### 3.1.7 偏置 bias

有时为了更好的拟合数据，会为输出的特征图加上一定偏置

因为涉及到对数据的拟合，pytorch支持对每一个输出通道设置偏置，并且这个偏置是可学习的，即在训练过程中会纳入计算图计算梯度

pytorch支持为每一个输出通道维护不同的偏置，存放在`torch.nn.Convxd`类下的`bias`张量中



#### 3.1.8 二维卷积示意

在`torch.nn`模块中，所有和二维卷积相关的操作都被组织在了`torch.nn.Conv2d`类中，该类的构造器为：

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',...)
```

这些参数的大部分都已经包括在了上面的说明中，这里给出一些附加说明

`kernel_size`就是**单个卷积核的尺寸**，可以是单个整数`k`，就是`k*k`的卷积核，也可以是一个长度为2的一维元组，代表卷积核的宽度与高度

同理，步长参数`stride`、空洞参数`dilation`、填充参数`padding`同样可以为单个整数或者一维元组。如果是元组，每个元素分别代表在各个维度的参数值

测试代码如下

```
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
```

测试代码输出`卷积后尺寸:  torch.Size([2, 508, 508])`并在`ch2/assets`相对路径下保存两张图片`conv.png`与`rand_conv.png`

![image-20250917202845884](assets\conv.png)

![image-20250917202845884](assets\rand_conv.png)

可以看出，边缘卷积核很好的提取了图像的边缘部分，而随机卷积核由于是在一定的`(-k,k)`范围内生成元素，本次输出了类似负片的特征图

#### 3.1.9 深度卷积

最后说一下和分组卷积联系紧密的深度卷积，在pytorch官方文档是这么描述它的（翻译后）

>当 `groups == in_channels` 且 `out_channels == K * in_channels`，其中 K 是一个正整数时，该操作也被称为“深度卷积”（depthwise convolution）。
>
>换句话说，对于一个尺寸为 `(N, C_in, H_in, W_in)` 的输入（注：原文写的是 L_in，但在图像中通常指 H_in × W_in），可以通过设置参数 `(in_channels=C_in, out_channels=C_in × K, ..., groups=C_in)` 来执行一个“深度乘子为 K”的深度卷积。 

从中可以得到，深度卷积就是**分组卷积的一种极端情况**：

- `groups = in_channels` ： 每个输入通道独立处理
- `out_channels = K * in_channels` ： 每个输入通道生成 K 个输出通道

也就是说：**每个输入通道，都用 K 个独立的卷积核进行卷积，生成 K 个输出通道。** 

其中，`K`被称为深度卷积的深度乘子

深度卷积是一种高效的卷积方式，**每个卷积核只处理一个输入通道**，常用于轻量级网络（如 MobileNet）

测试代码如下：

```python
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
```



上述测试代码会显示原图和rgb通道边缘卷积的的结果合并输出展示，并把rgb三个通道每个通道的卷积结果分别保存到`ch2/assets/edge_x.png`中

### 3.2 池化(Pooling)层

#### 3.2.1 基本概念

本部分参考了https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/Pooling.html

在图像处理中，由于图像中存在较多冗余信息，可用某一区域子块的统计信息（如最大值或均值等）来刻画该区域中所有像素点呈现的空间分布模式，以替代区域子块中所有像素点取值，这就是卷积神经网络中池化(pooling)操作。

引入池化的目的一般包括：

- **浓缩、约简图像特征**，但是保留主要信息，比如：当识别一张图像是否是人脸时，我们需要知道人脸左边有一只眼睛，右边也有一只眼睛，而不需要知道眼睛的精确位置，这时候通过池化某一片区域的像素点来得到总体统计特征会显得很有用
- 池化之后特征图会变小，缓解计算时的内存压力，并且对于后续的层次，能有效的**减少输入，节省存储空间并提高计算效率**。

池化操作也是通过**池化窗口**在输入图像上的滑动得到输出特征的。图池化窗口的大小也称为池化大小，用$k_h×k_w$表示。在卷积神经网络中用的比较多的是窗口大小为$2×2$，步幅为2的池化。

- 池化的主流用法是在任意第`i`维度，`stride_i>=k_i`，但是这不是硬性规定

池化的几种常见方法包括：平均池化、最大池化、K-max池化：

- **平均池化：** 计算区域子块所包含所有像素点的均值，将均值作为平均池化结果。
- **最大池化：** 从输入特征图的某个区域子块中选择值最大的像素点作为最大池化结果。对池化窗口覆盖区域内的像素取最大值，得到输出特征图的像素值。
- **K-max池化：** 对输入特征图的区域子块中像素点取前K个最大值，常用于自然语言处理中的文本特征提取。

![图1 平均池化和最大池化](assets\avgpooling_maxpooling.png)

![图2 K-max池化](assets\k-max_pooling.png)

#### 3.2.2 pytorch 类型

pytorch框架同样把池化操作抽象为了不同类型，包括一维至三维的最大值池化、平均值池化与自适应池化

| 层对应的类                     | 功能                                   |
| ------------------------------ | -------------------------------------- |
| `torch.nn.MaxPool1d()`         | 针对输入信号上应用 1D 最大值池化       |
| `torch.nn.MaxPool2d()`         | 针对输入信号上应用 2D 最大值池化       |
| `torch.nn.MaxPool3d()`         | 针对输入信号上应用 3D 最大值池化       |
| `torch.nn.MaxUnPool1d()`       | 1D 最大值池化的部分逆运算              |
| `torch.nn.MaxUnPool2d()`       | 2D 最大值池化的部分逆运算              |
| `torch.nn.MaxUnPool3d()`       | 3D 最大值池化的部分逆运算              |
| `torch.nn.AvgPool1d()`         | 针对输入信号上应用 1D 平均池化         |
| `torch.nn.AvgPool2d()`         | 针对输入信号上应用 2D 平均池化         |
| `torch.nn.AvgPool3d()`         | 针对输入信号上应用 3D 平均池化         |
| `torch.nn.AdaptiveMaxPool1d()` | 针对输入信号上应用 1D 自适应最大值池化 |
| `torch.nn.AdaptiveMaxPool2d()` | 针对输入信号上应用 2D 自适应最大值池化 |
| `torch.nn.AdaptiveMaxPool3d()` | 针对输入信号上应用 3D 自适应最大值池化 |
| `torch.nn.AdaptiveAvgPool1d()` | 针对输入信号上应用 1D 自适应平均池化   |
| `torch.nn.AdaptiveAvgPool2d()` | 针对输入信号上应用 2D 自适应平均池化   |
| `torch.nn.AdaptiveAvgPool3d()` | 针对输入信号上应用 3D 自适应平均池化   |

首先以自适应平均值池化为例介绍pytorch框架实现的自适应池化它不需要指定 `kernel_size`、`stride` 等参数，而是通过希望输出的大小，自动计算需要的池化窗口大小

`class torch.nn.AdaptiveAvgPool2d(output_size)`对由多个输入平面组成的输入信号应用二维自适应平均池化（Adaptive Average Pooling）。

它接受三维或者四维的输入：
$$
(N,C,H_{in},W_{in})\\(C,H_{in},W_{in})
$$

- 其中`N`代表批次大小，即一次传入了几张输入图像
- `C`代表传入图像的通道数
- `(H,W)`代表传入图像的尺寸

参数`output_size`表示期望的输出，可以为一维元组或单个整数，分别表示每个维度期望的输出尺寸或者期望正方形输出



然后介绍一下pytorch的普通池化功能

首先是**最大值池化**，以二维为例：

`class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`

其中大部分参数都已经在上面说明过，这里只介绍一些行为不同的

由于没有`padding_mode`参数，对象只会对输入图像按照`pad`规定的数量在滑动的末尾边缘添加零填充

`return_indices`参数的作用是让池化操作除了返回池化结果（即最大值），还返回 **最大值所在的位置索引**（需要设置变量进行接收）。这个索引是每个池化窗口中最大值在输入张量中对应的 **扁平化索引**。

> 扁平化索引，就是把张量按行优先方式（阅读的顺序）展开成一维张量后的索引

因此返回的索引张量是和输出特征图张量同形的



`ceil_mode`参数主要影响池化层对象在计算输出特征图尺寸时究竟是向上取整还是向下取整，这个参数对梯度更新或者优化影响很小，但是如果设置不当可能导致特征图在不同层次间无法对齐。（更细节的内容请参阅官方文档）



然后介绍**平均值池化**，同样以二维情况为例，pytorch类为

`class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)`

`divisor_override`参数让开发者定义求平均值时的分母，代替池化区域的元素个数。

`count_include_pad`参数决定池化层对象在池化区域包含零填充时是否将填充的零纳入平均值的计算。如果不纳入，分母会相应减少

- 如果设置了 `divisor_override`，分母会固定为`divisor_override` 值，`count_include_pad` 不再起作用



最后介绍一下表中提及的`MaxUnpoolxd`系列类型，同样是以二维为例：

`class torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)`

该类作为 `MaxPool2d` 的**逆操作（unpooling）**，通过记录最大池化时最大值的位置，将这些值恢复到原来的位置，从而实现某种意义上的上采样。它不会恢复被丢弃的非最大值，只是把最大值放回原来的位置，其余位置通常填充为 0 。

需要注意，在使用该类对象的`unpool(input,indices)`方法进行反卷积时，传入的`indices`参数通常需要是 `MaxPool`产生的索引张量 。



### 3.3 激活函数

开头的引入部分说激活函数需要避免梯度消失和梯度爆炸的问题，这里会介绍一些具体的激活函数，并说明它们的特点

本部分主要参考了

https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/activation_functions/Activation_Function.html#id2



#### 3.3.1 sigmoid

函数定义：

$$
f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

导数：

$$
f'(x) = f(x)(1 - f(x)
$$


![图3 sigmoid](assets\sigmoid.jpg)

优点：

1. sigmoid 函数的输出映射在 (0,1) 之间，单调连续，输出范围有限，优化稳定，可以用作输出层；
2. 求导容易；

缺点：

1. 由于其软饱和性，**一旦落入饱和区（离原点较远的区域）梯度就会接近于0**，根据反向传播的链式法则，容易产生梯度消失，导致训练出现问题；
2. Sigmoid函数的输出恒大于0。非零中心化的输出会使得其后一层的神经元的输入发生偏置偏移（Bias Shift），并进一步使得梯度下降的收敛速度变慢；
   - 关于偏置偏移的内容不难，同样在`ch2/extra/`文件夹中单独给出
3. 计算时，由于具有幂运算，计算复杂度较高，运算速度较慢。



#### 3.3.2 tanh

函数定义：

$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

导数：

$$
f'(x) = 1 - f(x)^2
$$


![图4 tanh](assets\tanh.jpg)

优点：

1. tanh 比 sigmoid 函数收敛速度更快；
2. 相比 sigmoid 函数，tanh 是以 0 为中心的；

缺点：

1. 与 sigmoid 函数相同，由于饱和性，**在远离原点时容易产生的梯度消失**；
2. 与 sigmoid 函数相同，由于具有幂运算，计算复杂度较高，运算速度较慢。



#### 3.3.3 ReLU

**Rectified Linear Unit**

函数定义：

$$
f(x) = 
\begin{cases}
0 & x < 0 \\
x & x \geq 0
\end{cases}
$$

导数：

$$
f'(x) = 
\begin{cases}
0 & x < 0 \\
1 & x \geq 0
\end{cases}
$$
![图5 ReLU](assets\relu.jpg)

优点：

1. 收敛速度快；
2. 相较于 sigmoid 和 tanh 中涉及了幂运算，导致计算复杂度高， ReLU可以更加简单的实现；
3. 当输入 x>=0 时，ReLU 的导数为常数，这样可有效缓解梯度消失问题；
4. 当 x<0 时，ReLU 的梯度总是 0，提供了神经网络的稀疏表达能力；

缺点：

1. ReLU 的输出不是以 0 为中心的；
2. 神经元坏死现象，某些神经元可能永远不会被激活，导致相应参数永远不会被更新；
3. 不能避免梯度爆炸问题；

神经元坏死指的是在神经网络训练过程中，有些神经元在整个训练阶段几乎 **不再激活**（输出恒为零或者接近零），它们对模型的输出没有贡献，就像“死掉”了一样。

> 相比于梯度消失，神经元坏死是一个更罕见的问题，因为它不仅需要激活函数在很长的区域内维持在接近零的位置

使用 **ReLU** 激活函数时，如果输入长期落在负区间，输出就是 **0**，梯度也为 **0**，该神经元就不会再更新参数 ，也就成为了 **坏死神经元**。



#### 3.3.4 LReLU

函数定义：

$$
f(x) = 
\begin{cases}
\alpha x & x < 0 \\
x & x \geq 0
\end{cases}
$$

导数：

$$
f'(x) = 
\begin{cases}
\alpha & x < 0 \\
1 & x \geq 0
\end{cases}
$$
![图6 LReLU](assets\lrelu.jpg)

优点：

1. 避免梯度消失；
2. 由于导数总是不为零，因此可减少死神经元的出现；

缺点：

1. LReLU 表现并不一定比 ReLU 好；
2. 无法避免梯度爆炸问题；



#### 3.3.5 PReLU

函数定义：

$$
f(\alpha, x) = 
\begin{cases}
\alpha & x < 0 \\
x & x \geq 0
\end{cases}
$$

导数：

$$
f(\alpha, x)' = 
\begin{cases}
\alpha & x < 0 \\
1 & x \geq 0
\end{cases}
$$
其中，$\alpha$也是一个可学习的参数



![图7 PReLU](assets\prelu.jpg)

PReLU 是 LReLU 的改进，可以自适应地从数据中学习参数；

收敛速度快、错误率低；

由于函数本身是可学习的，PReLU 可以用于反向传播的训练，可以与其他层同时优化；



#### 3.3.6 RReLU

函数定义：

$$
f(\alpha, x) = 
\begin{cases}
\alpha & x < 0 \\
x & x \geq 0
\end{cases}
$$

导数：

$$
f(\alpha, x)' = 
\begin{cases}
\alpha & x < 0 \\
1 & x \geq 0
\end{cases}
$$
其中 $\alpha$ 是一个服从特定区间的均匀分布的随机变量，pytorch框架允许用户自定义均匀分布的区间上下界



#### 3.3.7 SELU

(Scaled Exponential Linear Unit)

函数定义：

$$
f(\alpha, x) = \lambda
\begin{cases}
\alpha (e^x - 1) & x < 0 \\
x & x \geq 0
\end{cases}
$$

导数：

$$
f(\alpha, x)' = \lambda
\begin{cases}
\alpha e^x & x < 0 \\
1 & x \geq 0
\end{cases}
$$
其中 λ 和 α 是固定数值（分别为 1.0507 和 1.6726）

![图10 SELU](assets\selu.jpg)

优点：

1. **自归一化效果**：在合适条件下（比如输入数据标准化、使用 LeCun Normal 初始化、全连接深层网络），神经元的输出会自动保持在零均值、单位方差附近，不需要显式归一层。
2. **缓解梯度消失/爆炸**：因为输出分布会趋向稳定，深层网络在前向和反向传播中不容易梯度消失或爆炸
3. 保留负值



缺点：**条件限制强**，SELU 的自归一化效果依赖于：

- 网络是全连接前馈结构（FCN），在 CNN 或 RNN 中效果较差；
- 权重初始化要用 LeCun Normal；
- 输入需要标准化。
   如果这些条件不满足，自归一化优势会大打折扣。

SELU 的最大亮点是：在 **深层前馈全连接网络** 中，可以减少对 BatchNorm 归一层的依赖。但在现代主流架构（CNN、Transformer、RNN）中，由于其假设不成立，使用频率远低于 ReLU/GeLU/Swish。



#### 3.3.8 softsign

函数定义：

$$
f(x) = \frac{x}{|x| + 1}
$$

导数：

$$
f'(x) = \frac{1}{(1 + |x|)^2}
$$
![图11 softsign](assets\softsign.jpg)



优点：

1. **平滑性好**：相比 ReLU、硬饱和函数，Softsign 连续可微，梯度变化平滑，没有突然的拐点。
2. **输出范围有限**：输出被压缩在 $(-1, 1)$ 区间，类似 $\tanh$，能防止数值发散。
3. **能缓解梯度消失**：对大输入，Softsign 的渐近速度比 $\tanh$ 更慢（$\tanh(x)\to \pm1$ 指数级，Softsign 是 $1/|x|$ 级），这意味着在大输入下，梯度衰减得没有 $\tanh$ 那么快，缓解部分梯度消失问题。
4. **计算比 tanh 更便宜**：它只需要加法、除法和绝对值运算，没有指数运算，计算上更轻量。

缺点：相比于ReLU家族，**softsign激活函数的学习效率通常是更低的，或至少不具优势**



#### 3.3.9 softplus

函数定义：

$$
f(x) = \ln(1 + e^x)
$$

导数（导数是Sigmoid函数）：

$$
f'(x) = \frac{1}{1 + e^{-x}}
$$

![图12 softplus](assets\softplus.jpg)

它可以看作是是 ReLU 的一个光滑版本，没有 ReLU 的“硬拐点”。

- 当 $x \to -\infty$，$\text{Softplus}(x) \to 0$。
- 当 $x \to +\infty$，$\text{Softplus}(x) \to x$。

优点：

1. **梯度连续，不会突然消失**：它的导数在`(-inf, inf)`上存在且连续，并且始终大于零，不会出现 ReLU 的“死神经元”问题。
2. **保留非线性和稀疏性**：小于 0 的输入会被压到接近 0（但不是完全 0），这和 ReLU 类似，能保持部分稀疏性。



缺点：

1. **计算比 ReLU 慢**：需要计算 $\exp$ 和 $\log$，在大规模神经网络里比 ReLU 要贵一些
2. **梯度消失风险**
3. **稀疏性不如 ReLU 强**：因为对负输入输出并不是严格的 0，导致特征稀疏性下降。
   - 特征的稀疏性指的是针对特定的输入，只有部分神经元被激活的特性，稀疏特征让每个样本只激活一部分神经元，等于让不同的神经元“专门负责”不同模式。
   - 如果所有神经元都输出非零值，特征空间就很“密集”，样本之间容易互相干扰。稀疏化后，不同输入只会激活不同的少数神经元，使得不同类别在特征空间里更容易区分。

Softplus 激活函数的实际应用较少：在现代网络里很少作为默认激活函数使用，大多数情况下 ReLU/GELU/Swish 更常见



#### 3.3.10 softmax

**Softmax 激活函数** 是深度学习中非常重要的一个函数，尤其在**多分类问题**中被广泛使用。它常用于神经网络的**输出层**，将原始的“logits”（未归一化的分数）转换为一个**概率分布**，使得每个类别的输出值在 [0, 1] 之间，且所有类别的概率之和为 1。

> 在多分类任务中，由于有多种可能的决策，因此网络的输出个数通常与类别集合大小相等

softmax函数值可被解释为模型对每个类别预测的“置信度”或“近似概率”。它反映了在当前模型参数下，样本属于各个类别的相对可能性，但不一定是真实的统计概率。对于输出向量序列$z_1,z_2,...$
$$
P(y = i | \mathbf{x}) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$
其中$\mathbf{x}$为输入特征向量

示意图如下

![图13 三类分类问题的softmax输出示意图](assets\softmax.png)



#### 3.3.11 swish

**Swish 激活函数** 是由 Google 的研究团队在 2017 年提出的一种新型激活函数，它通过自动搜索（AutoML）的方式被发现，在多个深度学习任务中表现优于传统的 ReLU 激活函数。

swish 激活函数的形式非常简单
$$
Swish(x)=x*\sigma(\beta x)
$$
其中，

- $\sigma(\cdot)$为Sigmoid函数
- $\beta$为可学习参数或超参数，通常固定为1

![图16 swish 超参数](assets\swish2.jpg)

在$\beta=1$的情况下，swish 激活函数的一阶导数如下：

$$
\begin{aligned}
f'(x) &= \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) \\
&= \sigma(x) + x \cdot \sigma(x) - x \cdot \sigma(x)^2 \\
&= x \cdot \sigma(x) + \sigma(x)(1 - x \cdot \sigma(x)) \\
&= f(x) + \sigma(x)(1 - f(x))
\end{aligned}
$$
Swish 的形状类似于 ReLU，但它是**平滑且非单调**的。它在负值区域不会完全“死亡”（不像 ReLU 在 x<0 时输出为 0），而是保留一个很小的负值响应，这有助于缓解梯度消失和神经元“死亡”问题。

主要性质：

1. **平滑连续可导** ：有利于优化；
2. **非单调性** ： 在某些区域递减，有助于模型表达更复杂的函数；
3. **有界负值、无界正值** ： 类似于 Mish、GELU；
4. **自门控机制（Self-gating）** ： Sigmoid 部分可以看作对输入 x 的“门控权重”，实现自适应缩放；
5. **在深层网络中表现优异** ： 尤其在 ImageNet、ResNet、Transformer 等结构中超越 ReLU。



#### 3.3.12 h-swish

hard swish

h-swish 使用 **分段线性函数（ReLU6）** 来近似 Sigmoid，完全避免了指数运算，计算更快、更节能

h-swish 在 x ∈ [-3, 3] 区间内是线性的，两端饱和，形状与 Swish 非常相似，但在负值区域更“硬”。

![图17 Hard Swish](assets\hard_swish.jpg)



**性能接近 Swish**：在 MobileNetV3 等轻量模型中，精度损失极小，速度提升明显，但是**仅推荐用于轻量模型**：在服务器级大模型中，Swish 或 GELU 仍是首选。



#### 3.3.13 激活函数的选择

首先，对分类任务的**输出层：**

- **二分类** → 使用 `sigmoid`（输出概率在 [0,1]）
- **多分类（互斥）** → 使用 `softmax`（输出为类别概率分布）
- **多标签分类（可同时属于多个类）** → 使用多个 `sigmoid`（每个输出独立）

> 注意：`softmax` 通常用于最后一层，配合交叉熵损失；而 `sigmoid` 可用于多标签场景。 



**回归任务**的输出层一般使用：

- `linear`（无激活函数）或
- `softplus` / `softsign`（若需输出正数或平滑限制）

> 避免使用 `sigmoid` 或 `tanh`，除非你的目标值天然在 [0,1] 或 [-1,1] 区间。 



**隐藏层**激活函数选择：

| 激活函数                  | 推荐场景                   | 优点                           | 缺点                               |
| ------------------------- | -------------------------- | ------------------------------ | ---------------------------------- |
| **ReLU**                  | 大多数情况首选             | 计算快、缓解梯度消失、稀疏激活 | 死区问题（负输入为0）              |
| **Leaky ReLU**            | 存在负输入且希望避免死区   | 解决 ReLU 的死区问题           | 超参 α 需调参                      |
| **PReLU / RReLU**         | 数据量大时可用             | 自动学习负斜率，性能略优       | 更复杂，训练成本稍高               |
| **SELU**                  | 深层网络（如自编码器）     | 自归一化，稳定训练             | 对网络结构要求严格（需特定初始化） |
| **tanh**                  | 小型网络或旧模型           | 输出对称，适合某些传统架构     | 梯度饱和（易梯度消失）             |
| **sigmoid**               | 不推荐作隐藏层             | 平滑，但梯度小                 | 极易导致梯度消失                   |
| **softplus**/**softsign** | 需要平滑输出时             | 数学上连续可导                 | 比 ReLU 慢，效果不显著             |
| **Swish / h-swish**       | 新型网络（如 MobileNetV3） | 性能好，平滑                   | 计算开销稍大                       |

> 当前主流推荐： 
>
> - 隐藏层优先选 ReLU 或其变体（Leaky ReLU / PReLU）
> - 现代模型中 Swish 和 h-swish 表现优异，尤其在移动端或轻量级模型中



总结：

1. **优先考虑 ReLU 及其变体（Leaky ReLU / PReLU）**
   - 是大多数深度学习任务的默认选择。
   - 特别适用于 CNN、RNN、Transformer 的隐藏层。

2. **避免在隐藏层使用 sigmoid 和 tanh**
   - 它们容易导致梯度消失，尤其是在很深的网络中。
   - 除非有特殊需求（如旧模型复现、特定任务）。
3. **输出层必须匹配任务**
   - 分类 → softmax / sigmoid
   - 回归 → linear / softplus / softsign

4. **关注计算效率与硬件兼容性**
   - 如移动设备部署时，`h-swish` 比 `swish` 更高效（使用分段近似）。
   - `ReLU` 最快，适合实时推理。
5. **实验驱动选择**
   - 在相同架构下，尝试不同激活函数进行 A/B 测试。
   - 有时 `swish` 或 `PReLU` 会带来小幅提升。

### 3.4 循环层

 神经网络中的“循环层”通常指的是**循环神经网络（Recurrent Neural Network, RNN）**中的一类结构，常见的包括标准RNN、LSTM（长短期记忆）、GRU（门控循环单元）等。这类层特别适用于处理**序列数据**，比如时间序列、自然语言、语音信号等。

并且**在 RNN 等循环神经网络中，实际上并没有传统意义上神经元”的概念，取而代之的是一个经过设计的“细胞”整体，对数据进行处理，但是其仍旧遵循传统的线性变换 -> 非线性激活函数 -> 输出基础 的形式**

#### 3.4.1 循环层的主要功能

循环层可以处理任意长度的序列输入，（如一句话有不同数量的词），不像CNN那样需要固定维度。

循环层网络每个时间步依次处理一个元素（如一个单词或一个时间点的数据）。

通过状态传递保持一定程度上的“记忆”：循环层**通过隐藏状态（hidden state）在时间步之间传递信息**，使得模型能“记住”之前的信息。通过这种状态传递能力，**循环层能够建模序列中前后元素之间存在的依赖关系**（如语法结构、趋势变化等）

常见的循环层网络包括：标准RNN、LSTM和GRU

| 类型    | 特点                                                         | 适用场景                   |
| ------- | ------------------------------------------------------------ | -------------------------- |
| 标准RNN | 结构简单，但容易出现梯度消失/爆炸，难以学习长期依赖          | 短序列任务                 |
| LSTM    | 引入“细胞状态”和三个门（遗忘门、输入门、输出门），有效缓解长期依赖问题 | 长文本、复杂序列建模       |
| GRU     | LSTM的简化版，两个门（更新门、重置门），计算效率更高         | 中等长度序列，资源受限场景 |



#### 3.4.2 RNN的工作原理

RNN 的关键特点是 **“循环”（Recurrent）** —— 它在时间上共享参数，并通过一个隐藏状态（hidden state）传递信息。

假设我们有一个输入序列（比如一句话中的每一个字）：$\{x_1,x_2,...,x_T\}$，RNN 按时间步依次处理这些输入：

在时间步$t$，RNN接收当前输入 $x_t$ 和上一时刻的隐藏状态 $h_t$，计算出这一时刻的隐藏状态$h_t$，计算公式为：
$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$
可以看出，**隐状态的数学本质是一个一维向量**



此外，RNN还可以基于目前的隐状态 $h_t$ 对之后可能的输出进行预测：
$$
y_t = W_{yh} h_t + b_y
$$

上面式子中出现的各种 $W_{xy}$ 指的是层次 $y$ 接受层次 $ x $ 输入时的权重矩阵（或者反之，主要是看上下文语境），$b_h$ 、$b_y$ 都是偏置量


| 矩阵       | 含义                                                         | 维度说明（举例）                                             |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $ W_{xh} $ | 从输入（x）到隐藏层（h）的权重矩阵<br>负责将当前时刻的输入 $x_t$ 映射到隐藏状态空间 | 若输入维度是 $ d $，隐藏层大小是 $ h $，则 $ W_{xh} \in \mathbb{R}^{h \times d} $ |
| $ W_{hh} $ | 从上一时刻隐藏状态（h）到当前隐藏状态（h）的权重矩阵<br>实现“循环”机制，让模型记住历史信息 | $ W_{hh} \in \mathbb{R}^{h \times h} $                       |
| $ W_{yh} $ | 从隐藏层（h）到输出（y）的权重矩阵<br>将隐藏状态转换为输出结果 | 若输出维度是 $ o $，则 $ W_{yh} \in \mathbb{R}^{o \times h} $ |

$W_{xh} x_t$  相当于把当前输入$x_t$ （比如一个词向量）**投影到隐藏状态的空间中**

$W_{hh} h_{t-1}$ 把上一时刻的隐藏状态 $ h_{t-1} $ （即“记忆”）带入当前计算

在上述过程中，各种偏置量 $b$ 和权重矩阵 $W$ 都是可学习的



#### 3.4.3 LSTM的工作原理

LSTM（Long Short-Term Memory，长短期记忆网络）是一种特殊的循环神经网络（RNN），它被设计用来解决传统RNN在处理长序列数据时容易出现的**梯度消失**和**梯度过爆**问题。LSTM特别擅长捕捉时间序列中的长期依赖关系，在自然语言处理、语音识别、时间序列预测等领域有广泛应用。

LSTM 将传统RNN的隐状态`h_t`加工、包装为细胞状态`C_t`，通过在包装时的处理，实现了更优秀的表现，具体来说LSTM包含三个门：遗忘门、输入门、输出门。它们都使用Sigmoid函数（输出0~1）来决定哪些信息需要保留或丢弃。



**遗忘门（Forget Gate）**决定从细胞状态中**丢弃哪些信息**，也可以理解为**老状态在新状态中的权重参数**

公式：

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

- $ f_t $：遗忘门输出，值接近0表示“完全遗忘”，接近1表示“完全保留”
- $ h_{t-1} $：上一时刻的隐藏状态
- $ x_t $：当前时刻的输入
- $ [h_{t-1}, x_t] $ 表示将两个向量拼接
- $ W_f, b_f $：可训练的权重和偏置



**输入门（Input Gate）**决定**哪些新信息将被存储到细胞状态中**。

它包含两个部分：

- 一部分用Sigmoid判断哪些值需要更新
- 一部分用tanh生成新的候选值 $ \tilde{C}_t $

公式：

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

- $\tilde C_t$ 是**包含新输入的整个序列对应的候选状态**，或者说，“模型现在看到的信息，可能会成为新记忆。”
- $i_t$ 可以理解为状态 $\tilde C_t$ 在新状态中的权重参数



用遗忘门和输入门就能完成细**胞状态的更新**：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

其中 $ \odot $ 表示逐元素相乘

这一步实现了旧状态的部分遗忘和新信息的部分写入



**输出门（Output Gate）**决定细胞如何从当前的细胞状态 $C_t$ 输出隐藏状态 $h_t$ ，也即**更进一步地抽象了之后的细胞能从当前细胞状态中读取多少信息**

公式：

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

$ \tanh(C_t) $ 将细胞状态压缩到 $[-1, 1]$ 范围

最终 $h_t$ 是当前时间步的输出



#### 3.4.4 LSTM的优缺点

先说**优点**：

首先，**LSTM相比RNN，能有效捕捉长期依赖关系**，并在后续时间步中合理使用（例如：句子开头的主语在句尾才出现谓语）。

- 细胞状态通过加法操作更新（$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $），而不是乘法或非线性变换，因此梯度可以在反向传播时较为稳定地流动，这种结构避免了传统 RNN 中因连续矩阵相乘导致的梯度指数衰减问题
- 类比：普通 RNN 像电话游戏，消息传几轮就失真；LSTM 则像快递包裹，重要内容被封装保护，直到目的地才打开。



其次，**LSTM架构能缓解梯度消失/爆炸问题**，因为相比于传统RNN梯度会呈指数变化，而在 LSTM 中，细胞状态的梯度路径是线性传递为主的，即：

$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t
$$

当遗忘门输出接近 1 时，梯度几乎无损传递。加上门控机制对信息流的选择性控制，使得关键路径上的梯度不会快速衰减。



然后，**LSTM相比RNN具有更强的记忆选择能力（门控机制）**：三个门（遗忘门、输入门、输出门）均由 Sigmoid 函数控制，输出值在 [0,1] 区间，实现“软开关”功能。每个门都基于当前输入 $ x_t $ 和上一时刻隐藏状态 $ h_{t-1}$ 动态决策,具备上下文感知能力。



但是 LSTM 结构也有自身的缺点：

首先，**它计算复杂度高，并且不能并行化处理序列，导致训练速度很慢**。每个时间步包含 **4 个全连接层**（分别用于遗忘门、输入门、候选状态、输出门），参数量约为普通 RNN 的 3~4 倍。所有操作必须按时间顺序执行，无法并行化。GPU 并行优势难以发挥，尤其在长序列上性能瓶颈明显。

> 并行化处理是相对更先进的 Transformer 架构说的



而且**对于小型数据集或者序列较短的数据集**，它的参数多，**易过拟合**：每个门都有独立的权重矩阵（如 $ W_f, W_i, W_C, W_o $），总参数数量大，并且多层堆叠后参数量会呈指数增长。



**对于大型序列数据集或者序列较长的数据集，LSTM 的表示能力仍然有限，早期信息仍容易丢失**。尽管细胞状态理论上可长期保存信息，但实际中，遗忘门会逐渐衰减旧信息（即使 $ f_t \approx 0.99 $，经过 1000 步后也只剩 $ 0.99^{1000} \approx 4.3e^{-5} $），仍然无法彻底解决模型倾向于关注近期信息（recency bias）的问题。

> 这正是 **Transformer + Self-Attention** 能取代 LSTM 的关键原因之一



最后，它的可解释性由于多层包装也较差：门控值虽然是可观察的，但它们之间的交互过程高度复杂、非线性，导致不同任务下门的行为差异大，缺乏统一规律，使得很难准确判断某个预测结果是由哪段历史信息触发的。



由于这种“高不成低不就”的状态，目前 LSTM 在业界已经很少使用了

- 短序列使用 RNN 就能很好拟合
- 长序列被 Transformer 完全上位替代



#### 3.4.5 GRU 的工作原理

GRU 同样通过引入“门控机制”来更好地捕捉长期依赖关系，同时相比 LSTM（长短期记忆网络）结构更简单、计算效率更高。

GRU 引入了两个“门”来控制信息的流动：

1. **更新门（Update Gate）**
2. **重置门（Reset Gate）**

这两个门是可学习的，通过 sigmoid 函数生成介于 0 和 1 之间的值，用于决定保留多少历史信息和吸收多少新信息。

假设在时间步 $t$，输入为 $ x_t $，上一时刻的隐藏状态为 $ h_{t-1}$。

**重置门（Reset Gate）**$ r_t $ 的计算公式：
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$

- $ \sigma $ 是 sigmoid 函数，输出范围 $ (0,1) $
- $ W_r $ 是权重矩阵
- $ [h_{t-1}, x_t] $ 表示拼接操作

重置门的作用：决定上一时刻的隐藏状态 $ h_{t-1}$ 对当前候选状态的影响程度。如果 $ r_t \approx 0 $，则忽略过去信息。



计算重置门的作用是为了计算**候选隐藏状态** $ \tilde{h}_t $

计算公式为：
$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])
$$
候选隐藏状态可以看作是**前序隐藏状态通过遗忘门遗忘后**加上新的输入形成的隐藏状态



**更新门（Update Gate）**$ z_t $ 的计算公式：
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$
作用：决定当前状态 $h_t$ 中有多少来自过去 $ h_{t-1}$，有多少来自新信息。



计算更新门是为了给**隐藏状态的更新**提供权重参数，计算公式为：
$$
h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1}
$$
 这是一个加权组合：
  - $ z_t \odot h_{t-1} $：保留多少旧信息
  - $ (1 - z_t) \odot \tilde{h}_t $：加入多少新信息

当 $ z_t $ 接近 1 时，$ h_t \approx h_{t-1} $，即“记忆”长期状态，当 $ z_t $ 接近 0 时，$ h_t \approx \tilde{h}_t $，即“更新”为新状态



#### 3.4.6 GRU 的特点

相比 LSTM 少一个门，状态合并，减少了约 30% 的参数，训练更快，内存占用更低。

**在中等长度序列任务中表现优异**

但是**长期记忆能力略弱**：没有独立的细胞状态，所有信息都混合在 $h_t$ 中，可能导致远距离依赖信息被覆盖



#### 3.4.7 pytorch 类

在 PyTorch 中，提供了三种循环层以及它们单层的实现

| 层对应的类          | 功能                     |
| ------------------- | ------------------------ |
| `torch.nn.RNN`      | 多层 RNN 单元            |
| `torch.nn.LSTM`     | 多层长短时记忆 LSTM 单元 |
| `torch.nn.GRU`      | 多层门限循环 GRU 单元    |
| `torch.nn.RNNCell`  | 一个 RNN 循环层单元      |
| `torch.nn.LSTMCell` | 一个长短时记忆 LSTM 单元 |
| `torch.nn.GRUCell`  | 一个门限循环 GRU 单元    |



#### 3.4.8 RNN 相关类

对于`class torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)`类型，

`input_size` 参数需要为一个整数，含义是每个时间步输入向量的特征维度。 **它是原始数据经过预处理（如分词、嵌入、标准化、特征提取等）之后，输入到 RNN 每个时间步的向量维度。**

> 对于语言序列，常见的预处理流程为：原始文本 → 分词 → 单词索引 → 嵌入层（Embedding） → 向量序列 ，此时 `input_size = embedding_dim`（词向量维度）：如果用的是 128 维的词向量（如 `Word2Vec`, `GloVe`, 或可学习 Embedding），则 `input_size=128`；如果是 300 维（如预训练的 GloVe），则 `input_size=300`

由于这个特性，`input_size`必须与输入张量的最后一维，也即单个输入元的向量维度匹配



`hidden_size`参数也需要为一个整数，代表网络隐状态的向量维度，它通常是 `input_size` 的整数倍。

虽然单个 $h_t$ 是一维向量，但在实际使用中，PyTorch 处理的是**批处理（batch）和多个时间步**的数据，所以会组成更高维的张量。

**示例：**令

- `batch_size = 5`
- `seq_len = 10`
- `hidden_size = 64`

| 张量                               | 形状                         | 解释                |
| ---------------------------------- | ---------------------------- | ------------------- |
| 单个隐藏状态 $h_t$                 | `(64,)`                      | 一维向量            |
| 一批最后一个时间步的隐藏状态 $h_n$ | `(5, 64)`                    | batch 维度加入      |
| 所有时间步的输出`output`           | `(10, 5, 64)`或`(5, 10, 64)` | 时间 + batch + 特征 |



`num_layers`参数代表堆叠的 RNN 层数，默认为 1，在多层网络堆叠时，第一层的输入是预处理后的向量序列，而之后层次`l`（`l`大于等于2）的“输入序列”是第 `l−1` 层在每个时间步的输出（即隐藏状态），这些输出按时间顺序排列形成一个新的序列。 

因此数据流向可以用下图表示

```
Input (x_t)       Layer 1 RNN           Layer 2 RNN           Layer 3 RNN
    │         ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    ▼         │              │      │              │      │              │
[t=1] ──→ [RNN Cell L1] ──h1₁─┼─→ [RNN Cell L2] ──h2₁─┼─→ [RNN Cell L3] ──h3₁─→ output_1
    │         │              │  │   │              │  │   │              │
[t=2] ──→ [RNN Cell L1] ──h1₂─┼──┼→ [RNN Cell L2] ──h2₂─┼──┼→ [RNN Cell L3] ──h3₂─→ output_2
    │         │              │  │   │              │  │   │              │
  ...        ...            ... ... ...            ... ... ...            ...     ...
    │         │              │  │   │              │  │   │              │
[t=T] ──→ [RNN Cell L1] ──h1_T─┼──┼→ [RNN Cell L2] ──h2_T─┼──┼→ [RNN Cell L3] ──h3_T─→ output_T
              └──────────────┘  │   └──────────────┘  │   └──────────────┘

                ↓                 ↓                   ↓
           final_h[0]        final_h[1]          final_h[2]
```

多层 RNN 的主要作用是让**每层提取不同抽象程度的序列元素的内在联系**，从而增加网络整体对真实序列的拟合程度

而如果只使用一层时，它的作用和独立的`RNNRNNCell`没有数学上的差异



`nonlinearity`参数指定 RNN 网络内部使用的非线性激活函数，为字符串，可选 `'tanh'` 或 `'relu'`，默认 `'tanh'`。推荐仅使用 `'tanh'`，除非有强烈选用 `'relu'`的理由



`bias` 控制是否在权重矩阵中包含偏置项（bias），为布尔值，`True` / `False`，默认 `True`。

推荐保持 `True` 即可，有助于提升模型灵活性。但是如果有较高的性能需求，也可以设置为 `False`



下面先讲 `bidirectional` 参数，因为一些参数的理解要建立在这个参数的之上。它的含义是是否使用双向 RNN（Bi-RNN）

双向 RNN 指的是对输入序列正反向处理两次，**两次分别采用独立的权重矩阵、偏置值等可学习参数**，并将输出结果在最后一个维度上进行拼接（所以双向RNN的输出最后一维相较于单向RNN列数会加倍）

双向RNN的显著优点是能让 RNN 看到未来数据，相比于只能通过过去数据训练出的网络，显然对序列具有更强的拟合能力，尤其在NLP领域，语言是对一个抽象含义的序列化表述，语言序列本身上下文联系十分紧密，所以在NLP学科中双向 RNN 相比于单向更常用（但是现在都被Transformers替代了）

>考虑一句话：“我昨天买了苹果，很好吃。”，如果只从前向后读（单向），当读到“苹果”时，还不知道它是水果还是公司。如果也能从后往前看，“好吃”就提示了这是**水果**



`batch_first`参数影响网络如何理解输入张量，以及如何输出张量。

下面举个例子，假设：

- batch size = `B`
- sequence length = `L`，代表一个样本序列中**包含的时间步（time steps）或元素（elements）的数量。**
- input feature size = `D_in`，就是上面`input_size`的参数值
- hidden size = `H`

| 参数设置                    | 输入张量形状要求 | 输出张量形状含义             |
| --------------------------- | ---------------- | ---------------------------- |
| `batch_first=False`（默认） | `(L, B, D_in)`   | `(L, B, H * num_directions)` |
| `batch_first=True`          | `(B, L, D_in)`   | `(B, L, H * num_directions)` |

其中`num_directions` 就是RNN的方向数量，由上面的 `bidirectional` 参数控制

**`batch_first` 参数对模型的数学计算没有任何影响，只在工程/编程层面影响数据的组织方式。** 如果正在处理来自 `DataLoader` 的批次数据，强烈建议设置 `batch_first=True`，避免不必要的维度转换



最后，`dropout`参数规定了层与层之间的 dropout 概率，因此只在在`num_layers`参数值不为 1 时有效。 RNN 中的 dropout 是把低层向高层传递的隐状态向量中的元素随机置零。

> dropout为神经网络中的常用方法，同样在ch2/extra/dropout.md中进行介绍，也可以直接看https://zhuanlan.zhihu.com/p/38200980



此外，该类对象初始化后，还会自动初始化**一些重要的成员变量**（attributes）：

`weight_ih_l[k]` 和 `weight_hh_l[k]`

- **含义**：
  - `weight_ih_l[k]`: 第 `k` 层从 **输入到隐藏层** 的权重矩阵（input-to-hidden）
  - `weight_hh_l[k]`: 第 `k` 层从 **隐藏状态到隐藏层** 的权重矩阵（hidden-to-hidden）
- **形状**：
  - `weight_ih_l[k]`: 对于第0层为`(hidden_size, input_size)` ；对于深层是 `(hidden_size, num_directions * hidden_size)`
  - `weight_hh_l[k]`: `(hidden_size, hidden_size)`

 注意：当 `bidirectional=True` 时，每层有两个方向（正向和反向），所以会有： 

- `weight_ih_l[k]` 和 `weight_ih_l[k]_reverse`
- `weight_hh_l[k]` 和 `weight_hh_l[k]_reverse`



`bias_ih_l[k]` 和 `bias_hh_l[k]`

- **含义**：
  - 输入门和隐藏门的偏置项。
- **形状**：`(hidden_size,)`
- 只有当 `bias=True`（默认）时才存在。



 `all_weights`

- **类型**：`list` of `list`s
- **含义**：按层组织的所有权重名称（字符串列表），方便遍历
- 其结构如下所示

```
[
  ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'],
  ['weight_ih_l1', 'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1'],
  ...
]
```




#### 3.4.9 LSTM 相关类

`torch.nn`模块也是提供了多层LSTM网络和单层LSTM的实现，这里同样是介绍多层

`class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)`

该类构造器大部分参数都和 RNN 类的一致，只说明不一样的

.`proj_size` 参数会为每层添加一个可学习的投影矩阵 $W_{\text{proj}} \in \mathbb{R}^{p \times h}$ ，**附加的投影矩阵只改变 LSTM 的对外输出（即 `output` 和 `h_n`），而不会影响任何内部中间状态的计算过程。** 

实际用途

| 场景               | 作用                                                |
| ------------------ | --------------------------------------------------- |
| **减少参数量**     | 后续全连接层输入变小（比如从 512 → 128）            |
| **降低内存占用**   | `output`序列和`h_n`更小，利于长序列训练             |
| **控制模型宽度**   | 构建更深但不过宽的网络结构                          |
| **实现轻量化模型** | 在语音识别、机器翻译中广泛使用（如 Google 的 GNMT） |

> 需要注意的是，即使 `bias = True`，投影矩阵也不会附加可学习的偏置



此外，该类的各种权重矩阵、偏置变量也是可以直接访问，让用户自定义初始化或者直接读取的：

首先说明一下，pytorch对LSTM的实现和上文说的工作原理是一种数学计算的两种表示方式，在**计算门控权值时将前序隐藏状态和新输入用两个独立的的变换矩阵计算**而不是拼接后用一个矩阵变换，希望读者能建立这两种方法等价性的联系

对于任意时间步 $t$ 和第 $k$ 层 LSTM：
$$
i_t = \sigma(W_{ii}x_t + b_{ii} + W_{hi}h_{t-1} + b_{hi}) \quad \text{(输入门)}
$$

$$
f_t = \sigma(W_{if}x_t + b_{if} + W_{hf}h_{t-1} + b_{hf}) \quad \text{(遗忘门)}
$$

$$
g_t = \tanh(W_{ig}x_t + b_{ig} + W_{hg}h_{t-1} + b_{hg}) \quad \text{(候选状态)}
$$

$$
o_t = \sigma(W_{io}x_t + b_{io} + W_{ho}h_{t-1} + b_{ho}) \quad \text{(输出门)}
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t \quad \text{(细胞状态更新)}
$$

$$
h_t = o_t \odot \tanh(c_t) \quad \text{(隐藏状态输出)}
$$



`torch.nn.LSTM`类型将**和输入相关的变换矩阵打包进了一个变量`weight_ih_l[k]`**，其中 `k` 代表多层 LSTM 的层次，和输入门、遗忘门、候选状态、输出门的输入变换矩阵的数学名称为：$ W_{ii}, W_{if}, W_{ig}, W_{io} $
$$
\text{weight\_ih\_l}[k] = [W_{ii} \| W_{if} \| W_{ig} \| W_{io}] \in \mathbb{R}^{4 \times \text{hidden\_size} \times \text{input\_dim}(k)}
$$

其中，$\text{input\_dim}(k)$ 根据 $k$ 和 $\text{proj\_size}$ 的取值动态确定：

$$
\text{input\_dim}(k) =
\begin{cases}
\text{input\_size}, & \text{若 } k = 0 \\
\text{num\_directions} \times \text{hidden\_size}, & \text{若 } k > 0 \text{ 且 } \text{proj\_size} = 0 \\
\text{num\_directions} \times \text{proj\_size}, & \text{若 } k > 0 \text{ 且 } \text{proj\_size} > 0
\end{cases}
$$



同理，和前序隐藏状态相关的变换矩阵被打包进了一个变量 `weight_hh_l[k]`，其中 $k$ 代表多层 LSTM 的层次，对应输入门、遗忘门、候选状态、输出门的隐藏状态变换矩阵的数学名称为：$ W_{hi}, W_{hf}, W_{hg}, W_{ho} $。

$$
\text{weight\_hh\_l}[k] = [W_{hi} \,||\, W_{hf} \,||\, W_{hg} \,||\, W_{ho}] \in \mathbb{R}^{4 \times \text{hidden\_size} \times \text{hidden\_dim}(k)}
$$

其中，$\text{hidden\_dim}(k)$ 根据网络的层级结构和是否使用投影（$\text{proj\_size}$）动态确定：

$$
\text{hidden\_dim}(k) = 
\begin{cases}
\text{num\_directions} \times \text{hidden\_size}, & \text{若 } \text{proj\_size} = 0 \\
\text{num\_directions} \times \text{proj\_size}, & \text{若 } \text{proj\_size} > 0
\end{cases}
$$


对于输入变换的偏置项 `bias_ih_l[k]` 和前序隐藏状态变换的偏置项 `bias_hh_l[k]` 也是按照相同顺序打包，它们的形状就是$(4*hidden\_size,)$。有些实现（如 CuDNN）会将 `bias_ih` 和 `bias_hh` 合并成一个 `4*hidden_size * 2` 的向量，但在 PyTorch 接口中仍分开存储。



对于输出投影张量，每层的投影张量都存储在`bias_hh_l[k]`变量中，形状为`(4*hidden_size,)`



反向相关变量不赘述



最后说一下LSTM输入输出的和中间变量维度的对比

| 特性                 | RNN      | LSTM                    |
| -------------------- | -------- | ----------------------- |
| 输入维度             | 相同     | 相同                    |
| 输出`output`维度     | 相同     | 相同                    |
| 隐藏状态`hidden`结构 | 单一张量 | 元组`(h_n, c_n)`        |
| 是否需要初始化 `h0`  | 可选     | 可选                    |
| 是否需要初始化`c0`   | 不需要   | 可选（或自动初始化为0） |



#### 3.4.10 GRU 相关类

同理，先介绍`torch.nn.GRU`类型对工作原理的定义
$$
r_t = \sigma(W_{ir}x_t + b_{ir} + W_{hr}h_{(t-1)} + b_{hr})
$$

$$
z_t = \sigma(W_{iz}x_t + b_{iz} + W_{hz}h_{(t-1)} + b_{hz})
$$

$$
n_t = \tanh(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{(t-1)} + b_{hn}))
$$

$$
h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
$$

其中：

- $h_t$ 是时间步 $t$ 的隐藏状态
- $ x_t $ 是时间步 $t$ 的输入
- $ h_{(t-1)} $ 是该层在时间步 $ t-1 $ 的隐藏状态，或在时间步 0 的初始隐藏状态
- $ r_t $、$ z_t $、$ n_t $ 分别是重置门、更新门和新候选状态
- $ \sigma $ 是 sigmoid 函数，$ \odot $ 是 Hadamard 逐元素乘积

在多层 GRU 中，第 $ l $ 层（$ l \geq 2 $）的输入 $ x_t^{(l)} $ 是前一层在时间步 $t$ 的隐藏状态 $ h_t^{(l-1)} $，并乘以 dropout 噪声 $ \delta_t^{(l-1)} $，其中每个 $ \delta_t^{(l-1)} $ 是一个伯努利随机变量，以概率 `dropout` 取值为 0



然后看构造函数的定义：`class torch.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False,...)`

这些参数的行为和控制的内容都和之前介绍的基本一致，这里不赘述



对于输入，这里直接粘贴官方文档:

- `input`：张量，形状为 $(L, H_{in})$（非批量输入），或 $(L, N, H_{in})$（当 `batch_first=False` 时），或 $(N, L, H_{in})$（当 `batch_first=True` 时），包含输入序列的特征。
  - 输入也可以是打包后的可变长度序列，详情请参见 `torch.nn.utils.rnn.pack_padded_sequence()` 或 `torch.nn.utils.rnn.pack_sequence()`。（下一部分会简要说明）
-  `h_0`：张量，形状为 $(D \times \text{num\_layers}, H_{out})$ 或 $(D \times \text{num\_layers}, N, H_{out})$，包含输入序列的初始隐藏状态。若未提供，默认为零。

输出也是一样：

- `output`：张量，形状为 $(L, D \times H_{out})$（非批量输入），或 $(L, N, D \times H_{out})$（当 `batch_first=False` 时），或 $(N, L, D \times H_{out})$（当 `batch_first=True` 时），包含 GRU 最后一层在每个时间步 $t$ 的输出特征 $h_t$。如果输入是一个 `torch.nn.utils.rnn.PackedSequence`，则输出也将是一个打包序列。

- `h_n`：张量，形状为 $(D \times \text{num\_layers}, H_{out})$ 或 $(D \times \text{num\_layers}, N, H_{out})$，包含输入序列的最终隐藏状态。



#### 3.4.11 变长序列处理

在自然语言处理（NLP）中，**句子长度各不相同是常态**。如果你直接把不同长度的句子送入 RNN，会遇到维度不一致的问题。PyTorch 提供了一套完整的机制来高效地训练这种 **变长序列（variable-length sequences）** 数据集。

这里**主要讲原理，具体做法代码不需要掌握**，使用时学习即可

首先， 对于输入的许多序列，需要**分词**，**构建数据集的词表**，并将序列中的**单词映射到词表索引**

然后，pytorch提供了方法，可以**将索引序列用特定 `<PAD>` 值填充至相同长度**

> 在这个过程中**要记录有效序列的长度**，不然对每个序列都要在遍历时判断是否到达末尾不利于并发处理

最后，将索引序列集合**使用  `pack_padded_sequence` 打包**，打包后的对象能让 RNN 识别些是真实数据，哪些是填充的，避免 RNN 把 `<PAD>` 当作有效输入传播隐藏状态。

```python
#数据集例子
sentences = [
    "I love deep learning",
    "PyTorch is great",
    "RNNs are powerful for sequences",
    "Hi"
]
labels = [1, 1, 1, 0]  # 假设是情感分类标签

from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_sequence

# 简单分词
tokenized = [sent.lower().split() for sent in sentences]
# [['i', 'love', 'deep', 'learning'], ['pytorch', 'is', 'great'], ...]

# 构建词汇表
counter = Counter(word for sent in tokenized for word in sent)
vocab = {'<PAD>': 0, '<UNK>': 1}
vocab.update({word: idx + 2 for idx, word in enumerate(counter.keys())})

# 转换为索引序列
indexed_sentences = [[vocab.get(word, 1) for word in sent] for sent in tokenized]
# 示例：[[2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13, 14], [15]]

# 转成 tensor 并按长度排序（可选，提高 pack 效率）
seq_tensors = [torch.tensor(seq, dtype=torch.long) for seq in indexed_sentences]

# 获取原始长度
lengths = torch.tensor([len(seq) for seq in seq_tensors])

# 使用 pad_sequence 自动补齐
padded_seqs = pad_sequence(seq_tensors, batch_first=True, padding_value=0)
# 输出 shape: (batch_size, max_seq_len)

print(padded_seqs)

# 定义 embedding 层和 RNN
embed = nn.Embedding(vocab_size, embedding_dim=50)
rnn = nn.RNN(input_size=50, hidden_size=64, num_layers=1, batch_first=True)

# Embedding
embedded = embed(padded_seqs)  # (B, L, D)

# 打包：去除 padding 的影响
packed_embedded = pack_padded_sequence(
    embedded,
    lengths,
    batch_first=True,
    enforce_sorted=False  # PyTorch 1.11+ 支持非排序输入
)

# 输入 RNN
packed_output, hidden = rnn(packed_embedded)
```



#### 3.4.12 单层循环神经网络

`torch.nn` 模块还提供了单层的网络，主要用在堆叠方式高度自定义的网络中：

标准多层 RNN 是“一层接一层”的线性堆叠：

```
Layer1_output(t) → Layer2_input(t)
```

但用 Cell 可以实现：

- **跳跃连接（Skip connections）**：将第1层的输出直接传给第3层
- **残差连接（Residual connections）**：`h2 = cell2(x2, h1) + h1`
- **跨时间+跨层混合连接**
- 非顺序结构（如树形、图状传播）



堆叠Cell还能实现**每层的隐状态具有不同维度**：

```
layer1 = nn.LSTMCell(10, 64)
layer2 = nn.LSTMCell(64, 32)   # 更小
layer3 = nn.LSTMCell(32, 128)  # 再变大
```



**自定义时间步行为（非均匀/跳跃/重复）**



**混合不同类型单元**：可以在同一模型中自由组合：

- 第1层用 `LSTMCell`
- 第2层用 `GRUCell`
- 第3层用自定义 RNN 单元
- ……
