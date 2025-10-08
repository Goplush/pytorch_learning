[TOC]



# 4 数据操作与预处理

## 4.1 数据的加载

### 4.1.1 引入

在深度学习中，模型是心脏，而数据是血液。高效、灵活地管理和加载数据是成功训练模型的关键一步。PyTorch 提供了一套强大而优雅的工具来处理数据流程，其核心便是 `torch.utils.data` 模块中的 **Dataset**, **Sampler** 和 **DataLoader** 这三个类。

它们三者各司其职，协同工作，**将一个原始数据集转换成模型可以直接消化的小批量张量。**

打个比方，想象一个大型图书馆：

- **`Dataset`** 就像是整个**图书馆的书架**。它定义了数据的存储方式和总数，并且知道如何根据一个“索引”（如书名或索书号）来找到并取出一本具体的书（一个数据样本）。
- **`Sampler`** 就像是**图书管理员的取书策略**。他决定按什么顺序去书架上取书——是随机乱序地取，还是按顺序取，或者是根据某种特定规则（如书籍的重量）来取。
  - 需要注意的是`Sampler` 只产生“索引”（比如 [3, 7, 1]），它并不直接去书架上拿书。
  -  `Sampler` 只支持对可索引访问的数据集进行索引管理，而**对于只能迭代访问的数据集，由于访问方式单一，所以不需要`Sampler`**
- **`DataLoader`** 就像是**推着推车的助理**。他听从 Sampler 的指令，一次从书架上取回多本书（一个批次），并可能对这些书进行一些预处理，比如包上书皮（数据转换），然后把一整批书整齐地送到读者（模型）面前。

它们之间有着清晰的**合作工作逻辑**：`DataLoader` 是一个高级别的、面向用户的数据加载器。`DataLoader` 根据 `Dataset` 的类型采用不同的数据获取策略。对于支持随机访问的数据集，它会结合 `Sampler` 来定义数据读取顺序；对于流式数据集，则直接进行迭代读取。"

这种工作逻辑可以总结为下面的依赖图

```
┌─────────────┐
│   Dataset   │ ←── 存储数据，知道如何通过索引获取单个样本
└─────────────┘
       ↑
       │ 通过索引获取数据
       | 或者直接迭代数据流
       │
┌─────────────┐    ┌─────────────┐
│  DataLoader │ ←─ │   Sampler   │
└─────────────┘    └─────────────┘
       │                  │
       └──────────────────┘
    DataLoader 向 Sampler 请求索引，
    然后用这些索引从 Dataset 获取数据
   当数据集只能迭代访问时不需要Sampler
```



### 4.1.2 数据集

`Dataset` 是 PyTorch 数据加载管道（Data Loading Pipeline）的基石，**它是一个抽象概念，定义了组织和访问数据的接口**。因为现实场景下的数据集组织方式是不可预估的，因此 pytorch 框架没有实现统一的数据集访问接口，而是实现了不同种类的基类，让用户通过实现抽象方法，自主适配用到的数据集。

`torch.utils.data.Dataset` 是 PyTorch 数据加载管道（Data Loading Pipeline）的基石，所有的数据集类都应继承自这个基类或其子类，并实现特定的方法。

PyTorch 将 `Dataset` 分为两大范式：**映射式数据集**和**可迭代式数据集**，它们的设计哲学和适用场景截然不同。

#### 4.1.2.1 映射式数据集

首先是映射式数据集，这类数据集实现类需要直接继承 `torch.utils.data.Dataset` 类

并且需要实现 `__getitem__()` 和 `__len__()`方法。

- **`__getitem__(self, index)`**
  - **作用**: 这是 `Dataset` 类最核心的方法。它定义了如何根据给定的索引（或键）获取单个数据样本。
  - **输入**: `index` (通常是整数，但也可以是非整数的键)。
  - **输出**: 返回一个数据样本。这个样本通常是一个元组，**其具体结构完全由包装的数据集定义**，例如 `(data, label)` 或 `(input_tensor, target_tensor)`。
  - **行为**: 当你使用 `dataset[idx]` 语法时，实际上就是在调用这个方法。
  - **重要性**: 所有数据读取逻辑（如从文件加载图像、解析文本、访问数据库等）都应该在这个方法内部实现。
- **`__len__(self)`**
  - **作用**: 返回数据集的总样本数量。
  - **输入**: 无。
  - **输出**: 一个整数。
  - **重要性**: 许多 `Sampler`（采样器）和 `DataLoader` 的默认行为都依赖于这个方法来知道数据集的大小。

此外，由于目前将数据打包成批（batch）处理是很常用的方法，因此也推荐实现 `__getitems__(self, indices)` 方法，利用一个IO对象读取一批数据，充分利用python本身的IO优化

- **`__getitems__(self, indices)`**
  - **作用**: 这是一个可选的、用于**加速批量数据加载**的方法。当需要一次性获取多个样本时（例如，`DataLoader` 构造一个批次），框架会优先尝试调用此方法，而不是多次调用 `__getitem__`。
  - **输入**: `indices` (一个包含多个索引的列表)。
  - **输出**: 一个包含对应样本的列表。
  - **注意**: 如果没有实现 `__getitems__`，`DataLoader` 会自动降级为循环调用 `__getitem__`。

#### 4.1.2.2 可迭代对象与迭代器

在进入迭代式数据集的介绍之前，需要了解什么是可迭代对象与迭代器：

python 中的可迭代对象 (Iterable)就是实现了`__iter__()`方法的对象，`__iter__()`方法的功能是返回主调对象的迭代器

而迭代器的一般特点是只能向前移动，不能重置；遍历完成后，再次遍历需要重新创建迭代器；迭代器自身也是可迭代的对象

因此，迭代器需要实现两个方法：`__iter__()`和`__next__()`方法，`__next__()`方法的功能是返回下一个元素，如果没有更多元素则抛出`StopIteration`异常

一般情况下，可以为可迭代对象维护两个静态的迭代器——`start` 和 `end`， 这**两个迭代器都是不存在对应位置元素的抽象迭代器**。`start` 迭代器的下一个元素是首元素，`end` 迭代器对应的元素是末尾元素的下一个元素



相比于索引访问，这里介绍一种**惰性迭代器**——生成器，生成器是一种特殊的迭代器，使用函数和yield语句创建：

```python
def my_generator():
    yield 1
    yield 2
    yield 3

# 使用
gen = my_generator()
for value in gen:
    print(value)
```

**当调用生成器函数时，函数并不执行，而是返回一个生成器对象**。当第一次调用 `next()` 时，函数从开始处执行，直到遇到 `yield`，返回 yield 后面的值，并暂停。再次调用 `next()` 时，从上次暂停的位置继续执行，直到再次遇到 `yield`。如果遇到函数结束或 `return`，则抛出 `StopIteration` 异常。

生成器的优势

1. **节省内存**：对于大量数据，不需要一次性加载到内存。
2. **无限序列**：可以表示无限序列，如斐波那契数列。
3. **管道操作**：可以将多个生成器连接起来，形成处理管道。

使用生成器进行大文件读取：

```python
def read_lines(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# 使用生成器逐行读取大文件，避免内存不足
for line in read_lines('large_file.txt'):
    print(line)
```



#### 4.1.2.3 迭代式数据集

自定义的迭代式数据集类型需要继承 `torch.utils.data.IterableDataset`，并且实现 `__iter__()`方法。因此这也要求：

- 在该类的构造方法中，要将使用到的数据集包装为可迭代的对象。
- **提前实现该类的迭代器类**

由于迭代器的设计和数据特性关系很大，因此这里不涉及具体应该如何写迭代器

> 至于为什么说 `torch.utils.data.Dataset` 是所有数据集对象的基类，我们可以从源码中找到答案： `class IterableDataset(Dataset[_T_co], Iterable[_T_co]):`



#### 4.1.2.4 数据集的子集

`torch.utils.data.Subset` 类

```python
class torch.utils.data.Subset(dataset, indices)
```

由指定索引限制的数据集子集。

**参数**

- **dataset** (`Dataset`)：整个数据集。
- **indices** (`sequence`)：从整个数据集中选出用于子集的索引。



#### 4.1.2.5 包装张量的数据集

`torch.utils.data.TensorDataset` 类

```python
class torch.utils.data.TensorDataset(*tensors)
```

它将多个参数张量打包为一个数据集，这个方法不涉及对张量的拼接，**它是将张量统一用最高维索引**，因此要求参数张量的最高维列数相同

```python
import torch
from torch.utils.data import TensorDataset

X = torch.randn(100, 3, 32, 32)  # 100张 3x32x32 的图像
y = torch.randint(0, 10, (100,)) # 100个标签

dataset = TensorDataset(X, y)

print(len(dataset))  # 输出: 100
print(dataset[0])    # 输出: (X[0], y[0]) —— 第0个样本和其标签
```



#### 4.1.2.6 包装多个不同种类数据集

在一些场景中，我们需要将不同种类的数据一起训练，比如一些多模态模型的实现依赖将图片转化为文字描述，这种转化功能的训练集就需要包括图片和对应的描述

```python
class torch.utils.data.StackDataset(*args, **kwargs)
```

**参数**

- *args (Dataset) – 用于堆叠的数据集，以元组形式返回。
- **kwargs (Dataset) – 用于堆叠的数据集，以字典形式返回。

示例代码：

```python
images = ImageDataset()
texts = TextDataset()

# 元组形式堆叠
tuple_stack = StackDataset(images, texts)
assert tuple_stack[0] == (images[0], texts[0])

# 字典形式堆叠
dict_stack = StackDataset(image=images, text=texts)
assert dict_stack[0] == {'image': images[0], 'text': texts[0]}
```



#### 4.1.2.7 直接拼接数据集

`torch.utils.data.ChainDataset` 类

```python
class torch.utils.data.ChainDataset(datasets)
```

- **datasets** (`iterable` of `IterableDataset`)：要链接在一起的数据集。

`ConcatDataset` 的设计目的是**将多个数据集按“样本维度”（即第一个维度，样本数量）拼接起来**，形成一个更大的数据集。它**不要求每个子数据集的样本形状（shape）相同**，甚至不要求数据集中存放的数据类型相同

拼接后数据集的长度等于所有拼接数据集的长度之和

```python
import torch
from torch.utils.data import Dataset, ConcatDataset

class DS1(Dataset):
    def __len__(self):
        return 3
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32)  # 图像形状

class DS2(Dataset):
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        return torch.randn(10)         # 向量形状

ds = ConcatDataset([DS1(), DS2()])

print(ds[0].shape)  # torch.Size([3, 32, 32])
print(ds[4].shape)  # torch.Size([10]) —— 完全不同形状！
```



**模型输入通常要求固定形状，所以实践中，除非你有特殊处理逻辑，否则最好保持输入形状一致。**



#### 4.1.2.8 串联流式数据集

`torch.utils.data.ChainDataset` 是用于**串联多个 `IterableDataset`** 的工具，按顺序迭代所有子数据集中的样本，适用于**流式数据、无法随机访问的大数据场景**。

```python
class torch.utils.data.ChainDataset(datasets)
```

 

### 4.1.3 采样器

在 PyTorch 里，**取样器（Sampler）** 是 `torch.utils.data.Sampler` 的子类，用于控制 **Dataset 中样本的索引顺序**。它一般和 `DataLoader` 搭配使用，主要是为了 **灵活控制数据的取样方式**，比如随机打乱、分层采样、加权采样、分布式采样等。

首先介绍一下自定义迭代器的方法

每个 `Sampler` 子类必须提供 `__iter__()` 方法，该方法提供一种迭代数据集元素索引或索引列表（批次）的方式，并可以选择提供 `__len__()` 方法，返回迭代器每次迭代输出的长度（索引数目）。

而且，`Sampler` 对象的 `__iter__()` 方法**返回的是一个迭代器（iterator）**，而这个迭代器在被遍历时，会依次“产出”（yield）**单个索引或一批索引（即一个索引序列/列表）**。

之所以采用迭代器方式生成，是因为相比于一次性生成全部索引序列然后分批次返回，迭代器方式可以更灵活的控制采样逻辑：

- 随机打乱（`RandomSampler`）
- 按权重采样（`WeightedRandomSampler`）
- 分布式切分（`DistributedSampler`）

> 如果在获取索引序列的方法中实现这些功能，实际上和迭代器的工作模式没有本质区别了

**对于使用 `DataLoader` 的开发者来说，通常完全不需要关心“如何从`Sampler`中得到样本索引”这个底层细节。这确实是 `DataLoader` 对象的核心工作之一。**这也和pytorch 在设计数据读取时的**解耦**思想契合：`Sampler` 只负责“顺序”，`Dataset` 只负责“存储和读取”，`DataLoader` 负责将它们粘合在一起并处理多进程、内存锁定等复杂问题。这种解耦使得每个组件可以独立发展和替换。



下面介绍一些pytorch 提供的  `torch.utils.data.Sampler` 子类



#### 4.1.3.1 顺序采样器

`torch.utils.data.SequentialSampler` 类

```python
class torch.utils.data.SequentialSampler(data_source)
```

按顺序采样元素，始终以相同的顺序进行。

**参数**：

- **data_source** (`Dataset`)：要从中采样的数据集。



#### 4.1.3.2 随机采样器

`torch.utils.data.RandomSampler` 类

```python
class torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

随机采样元素。如果不放回（`replacement=False`），则从一个被打乱顺序的数据集中进行采样。如果放回（`replacement=True`），用户可以指定 `num_samples` 来决定抽取多少个样本。

**参数**

- **data_source** (`Dataset`)：要从中采样的数据集。
- **replacement** (`bool`, *可选*)：如果为 `True`，允许同一个索引被多次采样（有放回）。默认值为 `False`。
- **num_samples** (`int`, *可选*)：当 `replacement=True` 时，指定要抽取的样本总数。如果未指定，则默认等于 `len(data_source)`。
- **generator** (`Generator`, *可选*)：用于生成随机数的随机数生成器。



#### 4.1.3.3 从数据集特定子集中随机采样

`torch.utils.data.SubsetRandomSampler` 类

```python
class torch.utils.data.SubsetRandomSampler(indices, generator=None)
```

从给定的索引列表中随机地、无放回地采样元素。

**参数**

- **indices** (`sequence`)：一个索引序列，表示可以从哪些位置采样。
- **generator** (`Generator`)：采样过程中使用的随机数生成器。



#### 4.1.3.4 带权重随机取样

`torch.utils.data.WeightedRandomSampler` 类

```python
class torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
```

根据给定的概率（权重）从 `[0, ..., len(weights)-1]` 中采样元素，需要开发者保证权重序列的长度和数据集长度一致（否则为UB）。

**参数**

- **weights** (`sequence`)：一个权重序列，不需要总和为 1。
- **num_samples** (`int`)：要抽取的样本数量。
- **replacement** (`bool`)：
  - 如果为 `True`，样本是有放回抽取的。
  - 如果为 `False`，样本是无放回抽取的，即一旦某个索引被抽中，它在同一轮中不会再次被抽中。
- **generator** (`Generator`)：采样过程中使用的随机数生成器。



#### 4.1.3.5 批次采样器

`torch.utils.data.BatchSampler` 类

```
class torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
```

包装另一个采样器，以成批的方式产出索引。

每次迭代返回一个包含 `batch_size` 个索引的列表。

**参数**

- **sampler** (`Sampler`)：被包装的基础采样器，用于生成单个索引。
- **batch_size** (`int`)：每个批次包含的样本数量。
- **drop_last** (`bool`)：
  - 如果为 `True`，当最后一个批次不完整时，将其丢弃。
  - 如果为 `False`，保留最后一个不完整的批次。

#### 4.1.3.6 多卡训练取样器

（了解就行，用的时候再细学即可）

`torch.utils.data.distributed.DistributedSampler` 类

```python
class torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)	
```

它是 PyTorch 中用于在**多 GPU 分布式训练**场景下对数据集进行**分片采样**的工具类。它的主要目的是确保每个 GPU 进程只处理整个数据集的一个子集，从而实现数据并行训练，同时避免不同进程间的数据重复或遗漏。

- **自动划分数据集**：根据 `world_size` 和 `rank` 将数据均匀分配给各个进程
- **支持 shuffle**：每个 epoch 可打乱数据顺序（默认开启）
- **支持 drop_last**：决定是否丢弃最后不足一个 batch 的数据
- **与 DataLoader 配合使用**：作为 `sampler` 参数传入

**在每个 epoch 中，整个数据集都会被均分（或近似均分）到每个训练进程（每个 GPU）中，作为该进程的训练样本。**这是 `DistributedSampler` 的核心设计目标 —— **数据并行 + 无重复 + 全覆盖**。

因此，需要在每个 epoch 开始前调用 `sampler.set_epoch(epoch)`。如果不设置，每个 epoch 的数据划分顺序都一样（因为随机种子固定），模型可能无法充分泛化。`set_epoch` 会基于 epoch 改变随机种子，使每个 epoch 的 shuffle 不同。 

该采样器在与 `torch.nn.parallel.DistributedDataParallel` 配合使用时特别有用。在这种情况下，每个进程都可以将一个 `DistributedSampler` 实例作为 `DataLoader` 的 `sampler` 参数传入，并加载仅属于自己的那部分原始数据集。



### 4.1.4 数据加载器



```python
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='', in_order=True)
```

数据加载器（Data loader）将数据集和采样器（sampler）组合在一起，为给定的数据集提供一个可迭代对象。

`DataLoader` 支持对**映射式**和**可迭代式**数据集进行单进程或多进程数据加载，并支持自定义加载顺序、可选的自动批处理（collation）以及内存锁定功能。

下面给出一些需要了解的参数：

- **dataset** (`Dataset`)：用于加载数据的数据集对象。
- **batch_size** (`int`, *可选*)：每个批次加载多少个样本（默认值：`1`）。
- **shuffle** (`bool`, *可选*)：如果设为 `True`，则在每个 epoch 开始时重新打乱数据顺序（默认值：`False`）。
- **sampler** (`Sampler` 或 `Iterable`, *可选*)：定义从数据集中抽取样本的策略。可以是任何实现了 `__len__` 方法的可迭代对象。如果指定了此参数，则不能同时指定 `shuffle`。
- **batch_sampler** (`Sampler` 或 `Iterable`, *可选*)：类似于 `sampler`，但每次返回一批索引。与 `batch_size`、`shuffle`、`sampler` 和 `drop_last` 互斥。
- **num_workers** (`int`, *可选*)：用于数据加载的子进程数量。`0` 表示数据将在主进程中加载（默认值：`0`）。
- **collate_fn** (`Callable`, *可选*)：将样本列表合并成一个小批量张量的函数。在使用映射式数据集进行批处理加载时使用。



pytorch 框架并没有对`torch.utils.data.DataLoader` 类提供很多具有不同功能的子类。其核心原因是**鼓励用户通过组合和配置现有参数（如 `sampler`, `collate_fn`, `batch_sampler` 等）来满足各种需求。**

#### 4.1.4.1 epoch 和 batch

这里说明一下pytorch对于训练 epoch 和 batch 的关系。

首先是 epoch，它的含义是模型在训练集上完成一次完整遍历，也就是：

```
num_batches_per_epoch = len(dataset) // batch_size
```

模型会迭代 `num_batches_per_epoch` 次，使每个样本都被训练一次。

但是**在pytorch框架中，“遍历 dataloader 一次”通常就算一个 epoch**，但 dataloader 的采样方式可能意味着“并非严格遍历整个数据集”。例如有一个长度为5000的数据集，但是用如下的随机采样方式训练：

```python
DataLoader(dataset, sampler=WeightedRandomSampler(weights, num_samples=1000))
```

那么：

- 每个 epoch 只训练了 1000 个样本；
- 其中有的样本可能重复；
- 有的样本可能没出现；

所以这个 epoch **不是完整遍历整个数据集**。



### 4.1.5 整理函数 collate function

`torch.utils.data._utils.collate.collate` 函数

```python
torch.utils.data._utils.collate.collate(batch, *, collate_fn_map=None)
```

处理批处理内元素集合类型的通用整理（collate）函数。

该函数还开放了函数注册表，用于处理特定的元素类型。`default_collate_fn_map` 提供了对张量、NumPy 数组、数值和字符串的默认整理函数。

参数：

- **batch**：一个待整理的批次。
- **collate_fn_map** (`Optional[dict[Union[type, tuple[type, ...]], Callable]]`)：可选字典，将元素类型映射到相应的整理函数。如果元素类型不在该字典中，函数将按插入顺序遍历字典的每个键，如果元素类型是该键的子类，则调用相应的整理函数。



此外，pytorch 框架还提供了 `torch.utils.data.default_collate`函数，这个函数是 **`collate_fn=None`（默认值）时，`DataLoader` 会自动使用的默认整理函数 。**

```python
torch.utils.data.default_collate(batch)
```

接收一批数据，并将批内的元素放入一个额外外层维度为批次大小的新张量中。

输出的确切类型可能是 `torch.Tensor`、`torch.Tensor` 的序列、`torch.Tensor` 的集合，或保持不变，具体取决于输入类型。当 `DataLoader` 中定义了 `batch_size` 或 `batch_sampler` 时，此函数用作整理的默认函数。

以下是通用的输入类型（基于批内元素类型）到输出类型的映射：

- `torch.Tensor` → `torch.Tensor`（增加一个外层批次维度）
- NumPy 数组 → `torch.Tensor`
- float → `torch.Tensor`
- int → `torch.Tensor`
- str → str（保持不变）
- bytes → bytes（保持不变）
- `Mapping[K, V_i]` → `Mapping[K, default_collate([V_1, V_2, …])]`
- `NamedTuple[V1_i, V2_i, …]` → `NamedTuple[default_collate([V1_1, V1_2, …]), default_collate([V2_1, V2_2, …]), …]`
- `Sequence[V1_i, V2_i, …]` → `Sequence[default_collate([V1_1, V1_2, …]), default_collate([V2_1, V2_2, …]), …]`

**参数**

- **batch**：一个待整理的批次。

示例代码如下

```python
# 整数批次示例
default_collate([0, 1, 2, 3])
# 输出: tensor([0, 1, 2, 3])

# 字符串批次示例
default_collate(['a', 'b', 'c'])
# 输出: ['a', 'b', 'c']

# 批次内包含字典的示例
default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
# 输出: {'A': tensor([ 0, 100]), 'B': tensor([ 1, 100])}

# 批次内包含命名元组的示例
Point = namedtuple('Point', ['x', 'y'])
default_collate([Point(0, 0), Point(1, 1)])
# 输出: Point(x=tensor([0, 1]), y=tensor([0, 1]))

# 批次内包含元组的示例
default_collate([(0, 1), (2, 3)])
# 输出: [tensor([0, 2]), tensor([1, 3])]

# 批次内包含列表的示例
default_collate([[0, 1], [2, 3]])
# 输出: [tensor([0, 2]), tensor([1, 3])]

# 扩展 `default_collate` 以处理特定类型的两种方法
# 选项1：编写自定义整理函数并调用 `default_collate`
def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, CustomType): # 某些自定义条件
        return ...
    else: # 回退到 `default_collate`
        return default_collate(batch)

# 选项2：原地修改 `default_collate_fn_map`
def collate_customtype_fn(batch, *, collate_fn_map=None):
    return ...

default_collate_fn_map.update(CustomType, collate_customtype_fn)
default_collate(batch) # 自动处理 `CustomType`
```



## 4.2 数据的预处理

### 4.2.1 图像数据的预处理

在将图像输入到神经网络进行训练前，需要将图像处理成符合神经网络输入要求的张量，其中可能包括张量化、剪裁、压缩等步骤，这也被称为预处理。

实验已经证明，通过合适的预处理方法，可以突出图像或者其他数据和任务相关的特征，从而加快学习效率，提高学习成果。

`torchvision` 中的 `transform` 模块提供了多种图像处理操作，下面将主要的几种展示在下面

#### 4.2.1.1 图像转化为张量

 `transforms.ToTensor` 类型，其构造器为

```python
torchvision.transforms.ToTensor()
```

**参数说明**：

无显式参数。

**输入要求**：

- 类型：PIL Image（RGB/灰度/RGBA等）或 NumPy 数组 `[H, W, C]`，元素类型为 `uint8`，取值范围 `[0, 255]`。

**功能描述**：

将图像转换为 PyTorch Tensor，并执行以下操作：

1. 归一化：将像素值从 `[0, 255]` 映射到 `[0.0, 1.0]`；
2. 维度变换：将形状从 `[H, W, C]` 转换为 `[C, H, W]`（符合 PyTorch 的 channel-first 格式）；
3. 数据类型转换：转为 `torch.float32` 类型的张量。

**返回类型**：

`torch.FloatTensor` of shape `[C, H, W]`，数值范围 `[0.0, 1.0]`。



#### 4.2.1.2 图像标准归一化

`transforms.Normalize `类

构造器为：

```python
torchvision.transforms.Normalize(mean, std, inplace=False)
```

**参数说明**：

- **`mean`** (`sequence`): 每个通道的均值，格式为 `[M1, M2, ..., Mn]`，n 为通道数（如 RGB 图像是 `[0.485, 0.456, 0.406]`）。
- **`std`** (`sequence`): 每个通道的标准差，格式同上（如 `[0.229, 0.224, 0.225]`）。
- **`inplace`** (`bool`, default `False`): 是否就地修改输入张量（节省内存，但不可逆）。

标准化操作：

```
output[channel] = (input[channel] - mean[channel]) / std[channel]
```

该函数通过变换，**使每个通道的数据在维持原有分布类型的情况下，变成均值为 0、标准差为 1**

要求输入必须是 `ToTensor()` 后的结果，即形状为 `[C, H, W]` 且值在 `[0.0, 1.0]` 范围内的浮点张量。

它的目的，或者说优势是：

1. 加速训练 & 提高稳定性
   - 神经网络对输入特征的尺度敏感。如果不同通道或特征的分布差异很大（如一个通道均值是 100，另一个是 50），会导致梯度更新不稳定。
   - Normalize 后所有通道都具有相同的统计特性（均值 0，方差 1），有助于优化器更快收敛。
2. 匹配预训练模型的期望输入：大多数在 ImageNet 上预训练的模型，其训练时都会使用特定的标准化参数，如果你4要做迁移学习，就必须用**相同的 Normalize 参数**，否则输入分布不一致，模型性能会大幅下降。

测试代码：

```python
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
```



归一化以后的图像如下：

![image-20250924104941040](assets\lena_norm.png)

可以看出，归一化以后的图像虽然丢失了一部分原始特征，但是对比度更高，更有利于模型把握低维特征



#### 4.2.1.3 图像缩放

`transforms.Resize()`  类型

>  **注意**：`transforms.Scale()` 是早期版本中的名称，在较新版本的 PyTorch（>=1.4）中已被重命名为 `transforms.Resize()` 并标记为弃用。应使用 `Resize`。 

其构造器签名如下

```python
torchvision.transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None)
```

根据指定大小调整图像尺寸，支持保持宽高比或强制拉伸。

**参数说明**：

- **`size`** (`int` 或 `(height, width)` tuple):
  - 若为 `int`，则将图像的短边缩放到该值，保持宽高比；
  - 若为 `(H, W)` 元组，则直接调整为指定尺寸。
- **`interpolation`** (`InterpolationMode` 枚举类型，默认 `BILINEAR`): 插值方式，可选值包括：
  - `InterpolationMode.NEAREST`: 最近邻插值
  - `InterpolationMode.BILINEAR`: 双线性插值（默认）
  - `InterpolationMode.BICUBIC`: 双三次插值
  - `InterpolationMode.LANCZOS`: 高质量插值
- **`max_size`** (`int`, optional): 当 `size` 是整数时，限制长边最大长度。例如，若 `size=256`, `max_size=512`，则图像会被等比例缩放至短边为 256，但长边不超过 512。
- **`antialias`** (`bool`, optional): 是否启用抗锯齿（仅当 `interpolation` 支持时有效）。建议设为 `True` 提升画质。

`interpolation` 规定的插值方式不仅在上采样时发挥作用，下采样时也会发挥作用

- **上采样（放大，scale > 1）**
  - 插值决定“新像素”的填充方式。
  - 例如：
    - **NEAREST** → 复制最近的像素块，看起来像马赛克。
    - **BILINEAR** → 周围 4 个像素加权平均，更平滑。
    - **BICUBIC** → 周围 16 个像素加权，更细腻。
  - 上采样时，插值方法直接决定放大后的图像质感。
- **下采样（缩小，scale < 1）**
  - 插值决定如何“合并多个输入像素到一个输出像素”。
  - 举例：
    - **NEAREST** → 直接选取一个像素，可能丢失很多细节。
    - **BILINEAR** → 周围像素加权平均，更合理，但如果没有抗锯齿，高频细节可能混叠成条纹（aliasing）。
    - **BICUBIC** → 考虑更多邻域像素，效果比 bilinear 平滑一些。
  - 下采样时，插值决定哪些像素参与“压缩”，效果好坏体现在锯齿和细节丢失上。

下采样时，单纯用插值可能不足以避免混叠。`antialias=True` 会在缩小前对图像做一个低通滤波（模糊），去掉高频信息，再插值。这样能减少锯齿和摩尔纹。所以在 **缩小时推荐开 `antialias`**。



#### 4.2.1.4 中心剪裁

`transforms.CenterCrop` 类负责中心剪裁图像

构造器签名：

```python
torchvision.transforms.CenterCrop(size)
```

从图像中心裁剪出指定大小的子图，尽可能避免原始图像尺寸小于 `size` 参数规定的情况

**参数说明**：

- **`size`** (`int` 或 `(height, width)` tuple):
  - 如果是 `int`，裁剪出一个 `size × size` 的正方形区域；
  - 如果是 `(H, W)`，则裁剪高度为 H、宽度为 W 的矩形，中心与原图一致。

#### 4.2.1.5 普通随机剪裁

`transforms.RandomCrop` 类型可以实现一般的随机剪裁

其构造器签名为：

```python
torchvision.transforms.RandomCrop(
    size,
    padding=None,
    pad_if_needed=False,
    fill=0,
    padding_mode='constant'
)
```

随机选择裁剪位置，从中提取指定大小的图像块。常用于数据增强以增加多样性。

**参数说明**：

- **`size`** (`int` or tuple): 要裁剪的目标尺寸。
- **`padding`** (`int`, tuple, list, optional): 在四个边上添加的填充像素数。
  - 单个整数：四边都填充；
  - 四元组 `(left, top, right, bottom)`：分别设置各边填充。
- **`pad_if_needed`** (`bool`, default `False`): 如果图像比所需裁剪尺寸小，则自动填充使其足够大。
- **`fill`** (`int` or tuple): 填充值。
  - 对于单通道灰度图，是一个数字；
  - 对于三通道 RGB 图像，可以是 `(R, G, B)` 三元组。
- **`padding_mode`** (`str`, default `'constant'`): 填充模式，支持：
  - `'constant'`: 使用 `fill` 值填充；
  - `'edge'`: 复制边缘像素；
  - `'reflect'`: 镜像反射边界（不重复边缘点）；
  - `'symmetric'`: 对称反射（重复边缘点）。



#### 4.2.1.6 水平翻转

`transforms.RandomHorizontalFlip` 类可以将图像转化成的张量进行水平翻转

构造函数签名：

```python
torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

以概率 `p` 将图像沿垂直轴（左右）翻转。这是非常常见的数据增强手段，尤其适用于自然图像任务（如分类、检测），能提升模型泛化能力。

**参数说明**：

- **`p`** (`float`, default `0.5`): 执行水平翻转的概率。例如 `p=0.7` 表示有 70% 的概率执行翻转。



下面说明为什么垂直翻转能提升模型的泛化能力。

首先，**很多自然图像在语义上具有左右对称性**。在大多数自然场景中，物体本身和背景的 **语义信息左右对调后不会改变类别**。

- 猫从左看和从右看 → 依然是“猫”；
- 人站着向左或向右 → 依然是“人”；
- 房子、车、树等自然物体也大多对左右翻转不敏感。

因此，水平翻转不会改变任务标签，是“合法”的数据增强。

水平翻转在训练中可以 **提升模型的“方向不变性”**：

- 如果不用翻转，模型可能学到 **“猫脸必须朝左”** 才是猫。
- 加入水平翻转后，模型学到的就是 **“不管朝左还是朝右，都是猫”**，提升了泛化能力。
- 这种增强相当于人为地增加了一种 **平移不变性 / 旋转不变性**。

虽然常见，但也 **不是所有任务都能用**：

- **文字识别（OCR）**：左右翻转会把“b”变成“d”，语义就变了。
- **人脸关键点检测**：如果只翻图不翻 keypoints，标签会错乱。
- **医学影像**：某些影像的左右有明确语义（比如左肺 vs 右肺）。



#### 4.2.1.7 增强的随机剪裁

`transforms.RandomResizedCrop` 类型可以实现随机剪裁后缩放

其构造函数签名为

```python
torchvision.transforms.RandomResizedCrop(
    size,
    scale=(0.08, 1.0),
    ratio=(3./4., 4./3.),
    interpolation=InterpolationMode.BILINEAR,
    antialias=None
)
```

先在一个随机位置和尺度下裁剪图像（满足面积和宽高比约束），然后将其缩放到指定 `size`。广泛用于训练阶段的数据增强（如 ImageNet 训练常用此方法）。

**参数说明**：

- **`size`** (`int` or tuple): 输出图像的最终尺寸（如 `224` 或 `(224,224)`）。
- **`scale`** (`tuple`, default `(0.08, 1.0)`): 随机采样裁剪区域相对于原图面积的比例范围。例如 `(0.5, 1.0)` 表示裁剪部分至少占原图一半面积。
- **`ratio`** (`tuple`, default `(3/4, 4/3)`): 宽高比的范围。系统会在这个范围内随机选取一个宽高比来裁剪。
- **`interpolation`**: 同 `Resize`，默认双线性插值。
- **`antialias`**: 是否开启抗锯齿。



#### 4.2.1.8 图像填充

torchvision的transform模块提供了自己对于图像填充的实现： `transforms.Pad` 类

其构造函数签名为：

```python
torchvision.transforms.Pad(
    padding,
    fill=0,
    padding_mode='constant'
)
```

在图像四周添加指定数量的像素填充。可用于保证最小尺寸或防止后续操作越界。

**参数说明**：

- **`padding`** (`int` or tuple):
  - `int`: 四周均填充 `int` 个像素；
  - `(left, right, top, bottom)` 四元组：自定义每边填充量。
- **`fill`** (`int` or tuple): 填充值。
  - 单通道图像：标量值；
  - 多通道图像：`(R, G, B)` 形式的元组。
- **`padding_mode`** (`str`): 填充模式，同 `RandomCrop`，支持 `'constant'`, `'edge'`, `'reflect'`, `'symmetric'`。



#### 4.2.1.9 预处理组合

`transforms.Compose` 类可以把上述对图形的变换任意组合，形成一个序列，实现链式调用。常用于构建完整的数据预处理流程。

其构造器签名如下：

```python
torchvision.transforms.Compose(transforms)
```

**参数说明**：

- **`transforms`** (`list` of callables): 一个包含多个 transform 操作的列表，这些操作将按顺序依次应用于输入图像（通常是 PIL 图像或 Tensor）。

示例：

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```



#### 4.1.10 自定义预处理行为

torchvision 的 transform 模块还支持将用户自定义的操作打包，灵活扩展功能。

`transforms.Lambda` 类实现了这一功能

其构造函数签名如下：

```python
transforms.Lambda(lambd)
```

**参数说明**：

- **`lambd`** (`callable`): 任意可调用对象（函数、lambda 表达式等），接收一个输入（通常是 PIL 图像或 Tensor），并返回处理后的结果。



### 4.2.2 图像数据的导入

之前只介绍了抽象的数据集对象需要满足什么要求，实际上在torchvision的datasets模块中包含有 `ImageFolder` 类，该类型可以将如下格式的数据样本读取为数据集（每个子文件夹代表一个类别，子文件夹中的图像属于该类别）：

```
root/
  ├── class_1/
  │     ├── img1.jpg
  │     └── img2.png
  ├── class_2/
  │     ├── img3.jpg
  │     └── img4.jpeg
  └── ...
```



它的构造函数签名为：

```python
class torchvision.datasets.ImageFolder(
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Callable[[str], Any] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None
)
```

`root: str`

- **作用**：指定数据集的根目录路径
- 要求的目录结构如上
- 必须提供，且路径需存在并包含有效的图像文件和子目录

 `transform: Optional[Callable] = None`

- **类型**：可调用对象（通常是 `torchvision.transforms.Compose` 对象）
- **作用**：对加载的图像（PIL Image）进行预处理或增强。
  - 将预处理操作打包为一个方法的一个重要应用就是这个
- **应用时机**：在图像被读取后、返回前执行。

 `target_transform: Optional[Callable] = None`

- **类型**：可调用对象
- **作用**：对类别标签（`class_index`）进行变换。
- **应用场景**：
  - 将整数标签映射到 one-hot 编码（需自定义函数）
  - 标签偏移（如从 1 开始而不是 0）
  - 自定义编码方式

`loader: Callable[[str], Any] = None`

- **类型**：函数，接受一个字符串（文件路径），返回一个图像对象（通常为 PIL Image）
- **默认行为**：使用 `PIL.Image.open()` 加载图像
  - 通常不需要更改

 `is_valid_file: Optional[Callable[[str], bool]] = None`

- **类型**：函数，输入是文件路径，输出是布尔值（是否为有效文件）
- **作用**：决定某个文件是否应该被视为有效的图像样本。
- **优先级**：如果设置了 `is_valid_file`，则忽略文件扩展名检查；否则会检查是否是支持的图像格式。
- **默认支持的扩展名**（内部硬编码）：`IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')`

- **用途**：允许你自定义图像加载逻辑，例如：
  - 使用 OpenCV 加载（支持更多格式或颜色空间）
  - 添加异常处理
  - 支持非标准图像格式



它的使用效果为：

- **自动标注**：每个子文件夹的名字被视为一个类别，该文件夹内的所有图像都属于这个类别。
- **返回值**：每条数据是一个元组 `(image_tensor, class_index)`。
  - `image_tensor`: 经过 `transform` 处理后的张量。
  - `class_index`: 对应类别的整数标签（由 `dataset.class_to_idx` 可查）。



重要属性

| 属性                   | 说明                                             |
| ---------------------- | ------------------------------------------------ |
| `dataset.classes`      | 所有类别的名称列表，如`['cat', 'dog']`           |
| `dataset.class_to_idx` | 字典，映射类别名到索引，如`{'cat': 0, 'dog': 1}` |
| `dataset.imgs`         | 列表，包含所有`(image_path, class_index)`的元组  |



### 4.2.3 文本数据集的组织方式

1. **纯文本文件（.txt）**

- 每行一条样本。
- 常用于无标签任务（如语言模型）或简单分类任务。

```
This movie is great!	positive
I hate this film.	negative
...
```



2. **CSV/TSV 文件（.csv 或 .tsv）**

- 结构化存储，每行是一个样本，列对应字段（如文本、标签等）。
- 最常见的格式之一。

示例（CSV）：

```csv
text,label
"Great movie!",positive
"Terrible acting.",negative
```



3. **JSON/JSONL 文件**

- JSONL（JSON Lines）：每行是一个独立的 JSON 对象。
- 适合复杂结构数据。

示例（JSONL）：

```json
{"text": "I love it", "label": "positive"}
{"text": "It's boring", "label": "negative"}
```



4. **专用数据集格式（如 Hugging Face datasets）**



### 4.2.4文本数据集的常见预处理

#### 4.2.4.1 预处理流程

**加载数据**

- 从文件、内置数据集（IMDB、AG_NEWS 等）、或者自定义数据源读取原始文本。
- 数据通常是 (text, label) 或 (src, tgt) 这样的结构。

**分词 (Tokenization)**

- 把原始文本分割为词、子词或字符。
- 工具：torchtext 的 `get_tokenizer`，也可以用第三方分词器（如 spaCy, sentencepiece, HuggingFace Tokenizers）。

**构建词表 (Vocabulary)**

- 根据分词结果，统计词频并生成词表。
- 需要处理 OOV（未登录词），以及保留特殊符号（`<unk>`, `<pad>`, `<bos>`, `<eos>`, `<mask>`）。
- 工具：`torchtext.vocab.build_vocab_from_iterator` 或 `torchtext.vocab.Vocab`。

**数值化 (Numericalization)**：把分词结果映射为整数索引。

之前介绍的独热码就是在这里发挥作用的

- 例如：`["hello", "world"] → [123, 456]`。

**序列长度统一 (Padding/Truncation)**

- 因为句子长短不一，需要在 batch 里对齐。
- 常见做法：在 dataloader 的 `collate_fn` 中补齐 `<pad>`。

**构造 DataLoader**

- 把样本打包成 batch，得到张量输入。
- 工具：`torch.utils.data.DataLoader`，配合 torchtext 的 `Dataset`/`IterableDataset`。

**可选的进一步预处理**

- 子词切分（BPE、SentencePiece）
- 去除停用词
- 小写化、正规化

**数据增强**：为样本增加掩码用于训练，在NLP领域很常见



#### 4.2.4.2 torchtext 模块工具

在新版 torchtext（0.12+）里，推荐的核心组件主要是：

- **Tokenizer**
  - `torchtext.data.utils.get_tokenizer`
  - 支持 `basic_english`, `spacy`, `moses`, `sentencepiece` 等。
- **Vocabulary**
  - `torchtext.vocab.build_vocab_from_iterator`
  - `torchtext.vocab.Vocab`
  - 还可以用 `torchtext.vocab.GloVe`、`FastText` 等预训练词向量。



## 4.3 断点续训

深度学习中断点续训时，**Dataset**、**Sampler** 和 **DataLoader** 这三个组件需要保存的信息有所不同。

### 4.3.1 Dataset

`Dataset` **本身通常不需要保存额外状态**。因为：

- 它只是定义了数据的访问逻辑（如从文件或内存中读取样本）；
- 数据内容通常是静态的，不随训练过程变化。

只有以下情况才需要保存额外状态：

1. **动态数据集**（在线生成样本、数据增强随机性强）；
2. **可变数据源**（例如，训练期间更新的数据、或curriculum learning等自适应采样策略）。

在这种情况下，通常需要保存：

- 数据文件的加载进度（例如当前读到第几个样本）；
- 动态生成的随机种子或状态；
- 数据增强管线的随机状态（如果不是每次都固定种子）。



### 4.3.2 Sampler

**这是关键需要保存的组件**，特别对于 RandomSampler 和 WeightedRandomSampler。

常见的 `Sampler` 类型有：`RandomSampler`、`SequentialSampler`

不同类型的采样器所需状态不同：

| Sampler类型         | 需要保存的状态                                        |
| ------------------- | ----------------------------------------------------- |
| `SequentialSampler` | 当前样本索引位置（当前epoch中已取到第几个）           |
| `RandomSampler`     | 随机数生成器的状态（`torch.Generator().get_state()`） |
| 自定义Sampler       | 任意内部计数器、缓冲区等自定义状态                    |

这样恢复后你可以重新创建`Sampler`对象并：

```python
sampler.generator.set_state(saved_rng_state)
sampler.start_index = saved_start_index
```



### 4.3.3 DataLoader

`DataLoader` 本身通常是“无状态”的，但它**封装了多线程/多进程加载逻辑**，在断点续训时也有细节要处理。

关键点

1. `DataLoader` 不需要序列化（可以重新构造）；
2. 但如果你希望“从上次打断的 batch 开始继续”，则你需要记录：
   - 当前 epoch 编号；
   - 当前 batch 索引（在该 epoch 内的第几个 batch）；
   - 随机状态（尤其当 `shuffle=True` 时）。

