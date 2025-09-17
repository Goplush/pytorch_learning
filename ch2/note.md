[TOC]

https://docs.pytorch.org/docs/stable/search.html 这是Pytorch的文档搜索网址，可以方便的搜索到本文绝大部分方法和类型的文档

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

![图5 步幅为2的卷积过程](D:\data\repos\pytorch_learning\ch2\assets\stride.png)



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

![图7 感受野为5×5的卷积](D:\data\repos\pytorch_learning\ch2\assets\Receptive_Field_5x5.png)

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

![图9 多输出通道计算过程](D:\data\repos\pytorch_learning\ch2\assets\multi_out_channel.png)



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

- `batch`：每批有多少张图片



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

可以看出，边缘卷积核很好的提取了图像的边缘部分，而随机卷积核由于是在一定的`(-k,k)`范围内，本次输出了类似负片的特征图

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

