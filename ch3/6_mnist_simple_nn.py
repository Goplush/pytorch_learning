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
    train_dataset = datasets.MNIST(root='./ch3/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 验证集（从训练集中划分，或使用原始测试集作为验证）
    val_dataset = datasets.MNIST(root='./ch3/data', train=False, download=True, transform=transform)
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