#导入所需的库
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


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

class MyNet2(nn.Module):
    def __init__(self):
        super().__init__()
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
model = MyNet()
model = MyNet2()

# 定义一个简单的全连接网络
model = nn.Sequential(
    nn.Linear(784, 128),    # 输入784维，输出128维
    nn.ReLU(),              # ReLU激活函数
    nn.Linear(128, 64),     # 输入128维，输出64维
    nn.ReLU(),              # ReLU激活函数
    nn.Linear(64, 10)       # 输出10类（如MNIST分类）
)

#使用 OrderedDict 定义网络
model = nn.Sequential(OrderedDict([
    ('fc_layer_1', nn.Linear(784, 128)),
    ('relu_layer_1', nn.ReLU()),
    ('fc_layer_2', nn.Linear(128, 64)),
    ('relu_layer_2', nn.ReLU()),
    ('output', nn.Linear(64, 10))
]))
print(model)

