import torch
import numpy as np

def dtype_test():
    #declare a tensor
    t_a=torch.tensor([1.2,3.4])
    #get it's type
    print("data type of tensor {t_a} is {dtype}".format(t_a=t_a,dtype=t_a.dtype))


def change_dtype():
    # 设置默认张量类型为 DoubleTensor
    torch.set_default_dtype(torch.double)
    print("默认张量类型:", torch.tensor([1.2, 3.4]).dtype)
    

    # 将张量数据类型转化为整型
    a = torch.tensor([1.2, 3.4])
    print("a.dtype:", a.dtype)
    print("a.long() 方法:", a.long().dtype)
    print("a.int() 方法:", a.int().dtype)
    print("a.float() 方法:", a.float().dtype)


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
    print("tensor对象的size()方法:",tensor_4d.size())



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


def build_tensor():
    a=torch.tensor([[2,3],[4,5]],dtype=torch.float64)
    b=torch.Tensor([[2,3],[4,5]])
    c=torch.Tensor(2,3,3)
   #上一行等价于d=torch.Tensor(torch.Size([2,3,3]))

    print("torch.tensor()方法通过列表创建的张量:",a,"; 其数据类型为:",a.dtype)
    print("torch.Tensor()构造方法通过列表创建的张量:",b,"; 其不支持指定数据类型，因此构造出张量内存放的数据类型为",b.dtype)
    print("torch.Tensor()构造方法构造的指定形状为2*3*3的张量:",c)

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
    print("用new_empty方法生成的2*3*3未初始化张量:",d)

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

def gen_uniform_tensor():
    template=torch.Tensor(2,3)
    a=torch.rand(2,3)
    b=torch.rand_like(template)

    print("torch.randn() 方法生成的元素服从[0,1)均匀分布的2*3张量:\n",a)
    print("torch.randn_like()方法生成的和\n",template,"\n同形的，元素服从[0,1)均匀分布的张量:\n",b)

def shuffle_0_to_n(n:int):
    a=torch.randperm(n)
    
    print("利用torch.randperm将0 到 ",n,"-1 的所有整数打乱:",a)

def static_special_tensor_gen():
    zeros_tensor = torch.zeros(2, 3)  # 2x3 的全0张量
    ones_tensor = torch.ones(2,3)
    id_tensor = torch.eye(3)  # 3x3 单位矩阵
    full_tensor = torch.full((2, 3), 3.14)  # 2x3，用 3.14 填充
    empty_tensor = torch.empty(2, 3)  # 2x3 未初始化张量

    print("torch.zeros()方法生成的2*3全零张量:\n",zeros_tensor)
    print("\ntorch.ones()方法生成的2*3全 1 张量:\n",ones_tensor)
    print("\ntorch.eye()方法生成的3阶单位向量:\n",id_tensor)
    print("\ntorch.full()方法生成的2*3张量,用3.14填充:\n",full_tensor)
    print("\ntorch.empty()方法生成的2*3 空张量:\n**由于未初始化，每次生成的内容可能不一样**\n",empty_tensor)


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



if __name__ == "__main__":
    #dtype_test()
    #change_dtype()
    #tensor_shape()
    #tensor_ele_num()
    #build_tensor()
    #make_alike_tensor()
    #new_xx_series()
    #gen_normal_distribution_tensor()
    #gen_uniform_tensor()
    #shuffle_0_to_n(8)
    #static_special_tensor_gen()
    tensor_range_ops()