import torch
import numpy as np

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

def squeeze_tensor():
    x = torch.zeros(2, 1, 2, 1, 2)
    print("the shape of x is:",x.size())
    y=torch.squeeze(x,3)
    print("after squeeze the 3rd dim, the shape is:",y.size())
    y=torch.unsqueeze(y,3)
    print("unsing unsqueeze to put one col back to the 3rd dim, the shapeis:",x.size())

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

def tri_torch():
    a = torch.randn(4, 4)
    print("张量a:\n",a)
    b=torch.tril(a,0)
    print("\na的下三角矩阵形式:\n",b)
    c=torch.triu(a,1)
    print("\na的上三角矩阵形式，主对角线上方多去一条:\n",c)

def joint_tensor():
    a=torch.arange(6.0).reshape(2,3)
    print("第一个张量a:\n",a)
    b=torch.randn_like(a)
    print("\n第二个张量b:\n",b)

    c=torch.cat((a,b),dim=1)
    print("\na,b在第1维度（列维度）拼接后的向量:\n",c)

    d=torch.stack((a,b),dim=2)
    print("\na,b在第2维度（新维度）拼接后的张量:\n",d)

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


if __name__ == '__main__':
    #grad_test()
    #np2tensor2np()
    #tensor_reshape()
    squeeze_tensor()
    #tensor_slice()
    #tri_torch()
    #joint_tensor()
    #split_tensor()
