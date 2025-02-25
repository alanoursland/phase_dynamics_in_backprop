import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import random
import numpy as np
from torch.optim import Adam
from vadam import Vadam

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = 10

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Updated dataset loading function to support CIFAR-10
def load_dataset(name='MNIST', data_path='e:/ml_datasets'):
    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    elif name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f'Unknown dataset: {name}')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Neural network model class
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

# Updated model class for CIFAR-10 (simple CNN architecture)
class CIFAR10Model(nn.Module):
    def __init__(self, activation_fn=nn.ReLU()):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = activation_fn

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Update model and optimizer initialization for CIFAR-10
def initialize_models_and_optimizers(optimizer_name='Adam', dataset_name='MNIST'):
    activations = {'ReLU': nn.ReLU(), 'Abs': lambda x: torch.abs(x), 'Square': lambda x: x ** 2}
    
    if dataset_name == 'MNIST':
        models = {name: SimpleModel(activation_fn=act).to(device) for name, act in activations.items()}
    elif dataset_name == 'CIFAR10':
        models = {name: CIFAR10Model(activation_fn=act).to(device) for name, act in activations.items()}
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    if optimizer_name == 'Adam':
        optimizers = {name: Adam(model.parameters(), lr=learning_rate) for name, model in models.items()}
    elif optimizer_name == 'Vadam':
        optimizers = {name: Vadam(model.parameters(), lr=learning_rate) for name, model in models.items()}
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    return models, optimizers

# Training function
def train(models, optimizers, train_loader, epoch, criterion):
    for model in models.values():
        model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Testing function
def test(model, loader, criterion):
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

# Experiment runner with result saving
def run_experiment(dataset_name='MNIST', optimizer_name='Adam', experiment_name='default_experiment', num_runs=5):
    print(f"Running {experiment_name} for {num_runs} runs")
    for run_id in range(num_runs):
        print(f"Starting run {run_id}")
        seed = random.randint(0, 2**32 - 1)
        set_random_seeds(seed)
        train_loader, test_loader = load_dataset(dataset_name)
        models, optimizers = initialize_models_and_optimizers(optimizer_name, dataset_name)
        criterion = nn.CrossEntropyLoss()

        # Prepare output directories
        output_dir = f'./results/{experiment_name}/run_{run_id}/'
        os.makedirs(output_dir, exist_ok=True)

        # Save experiment metadata
        metadata = {
            'dataset': dataset_name,
            'optimizer': optimizer_name,
            'experiment_name': experiment_name,
            'run_id': run_id,
            'seed': seed,
            'epochs': epochs
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # Run training
        run_results = []
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train(models, optimizers, train_loader, epoch, criterion)
            epoch_results = {'epoch': epoch, 'time': time.time() - start_time, 'results': {}}

            # Create a string to display results for all models
            models_results = []

            for name, model in models.items():
                train_loss, train_acc = test(model, train_loader, criterion)
                test_loss, test_acc = test(model, test_loader, criterion)
                epoch_results['results'][name] = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }

                # Format the result string for this model
                model_result = f"{name}({train_loss:.3f}, {train_acc:.2f}%, {test_acc:.2f}%)"
                models_results.append(model_result)

                # Save model
                torch.save(model.state_dict(), os.path.join(output_dir, f'{name}_model_epoch_{epoch}.pt'))

            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch}: {'; '.join(models_results)} [Time: {elapsed_time:.2f}s]")

            run_results.append(epoch_results)

        # Save epoch results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(run_results, f, indent=4)
