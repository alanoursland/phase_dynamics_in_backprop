import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from adam import Adam

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = 10

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('e:/ml_datasets', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('e:/ml_datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Neural network models
class SimpleModel(nn.Module):
    def __init__(self, activation_fn=nn.ReLU()):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.activation = activation_fn
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Initialize models with different activations
model_relu = SimpleModel(activation_fn=nn.ReLU()).to(device)
model_abs = SimpleModel(activation_fn=lambda x: torch.abs(x)).to(device)
model_square = SimpleModel(activation_fn=lambda x: x ** 2).to(device)

# Optimizers
optimizer_relu = Adam(model_relu.parameters(), lr=learning_rate)
optimizer_abs = Adam(model_abs.parameters(), lr=learning_rate)
optimizer_square = Adam(model_square.parameters(), lr=learning_rate)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training and testing functions
def train(models, optimizers, train_loader, epoch):
    for model in models:
        model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def test(model, loader):
    model.eval()
    correct = 0
    loss_total = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_total += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(loader.dataset)
    return loss_total / len(loader), accuracy

# Main training loop
models = [model_relu, model_abs, model_square]
optimizers = [optimizer_relu, optimizer_abs, optimizer_square]
activation_names = ['ReLU', 'Abs', 'Square']

for epoch in range(1, epochs + 1):
    start_time = time.time()
    train(models, optimizers, train_loader, epoch)
    results = []
    for name, model in zip(activation_names, models):
        train_loss, train_acc = test(model, train_loader)
        test_loss, test_acc = test(model, test_loader)
        results.append(f'{name}({train_loss:.3f}, {train_acc:.2f}%, {test_acc:.2f}%)')
    elapsed_time = time.time() - start_time
    print(f'Epoch {epoch}: ' + '; '.join(results) + f' [Time: {elapsed_time:.2f}s]')
