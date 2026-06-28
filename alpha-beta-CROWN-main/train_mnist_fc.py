import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# 超参数
batch_size = 128
epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# 定义网络结构（示例：2层，每层256）
def create_model(num_layers=2, hidden_size=256):
    layers = []
    layers.append(nn.Flatten())
    layers.append(nn.Linear(28*28, hidden_size))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, 10))
    return nn.Sequential(*layers)

# 训练函数
def train_model(model, epochs=epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")
    print("训练完成！")

# 保存模型
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"模型已保存至 {filename}")

# 主程序
if __name__ == "__main__":
    # 你可以修改这里的层数
    configs = [
        (2, 256, "mnist_fc2_256.pt"),
        (4, 256, "mnist_fc4_256.pt"),
        (6, 256, "mnist_fc6_256.pt")
    ]
    for layers, hidden, fname in configs:
        print(f"\n正在训练 {layers} 层网络，隐藏层大小 {hidden}...")
        model = create_model(layers, hidden).to(device)
        train_model(model)
        save_model(model, fname)