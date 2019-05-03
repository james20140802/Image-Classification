from __future__ import print_function
from __future__ import division

from EnsembleNet import EnsembleNet
from EnsembleNet import ResidualBlock
from EnsembleNet import BasicBlock

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size = 128
epochs = 1000

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = CIFAR10('./data', train=True, download=True, transform=transform)
val_data, train_data = torch.utils.data.random_split(train_data, [5000, 45000])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

test_data = CIFAR10('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EnsembleNet(ResidualBlock, [3, 4, 6, 3], planes=64, num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0001)

model.to(device)


def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        optimizer.zero_grad()

        loss = criterion(output, target)
        total_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = F.softmax(output, 1).data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        loss.backward()
        optimizer.step()
    total_loss /= len(train_loader.dataset)
    print('Epoch: {} Training Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        (epoch+1), total_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def evaluate():
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        val_loss += F.cross_entropy(output, target, size_average=False).item()

        pred = F.softmax(output, 1).data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    val_loss /= len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(epochs):
    train(epoch)
    evaluate()

torch.save(model, 'cifar10.pt')
