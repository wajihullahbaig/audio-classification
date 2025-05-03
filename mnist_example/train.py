# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:56:29 2024

@author: Acer
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import seaborn as sns

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

class Softpick(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x):
        mmax = torch.max(x, dim=self.dim, keepdim=True)
        m = mmax.values
        num = torch.exp(x-m) - torch.exp(-m)
        numer = torch.relu(num)
        d = torch.abs(num)
        denom = torch.sum(d, dim=self.dim, keepdim=True) + self.eps
        return numer/denom

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        #self.soft_func = Softpick(dim=1, eps=1e-6)
        self.soft_func = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.soft_func(logits)
        return predictions

def download_mnist_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=transform
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=transform
    )
    return train_data, validation_data

def create_data_loader(data, batch_size):
    return DataLoader(data, batch_size=batch_size)

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    losses = []
    accuracies = []
    f1_scores = []
    precisions = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    for input, target in progress_bar:
        input, target = input.to(device), target.to(device)
        
        prediction = model(input)
        loss = loss_fn(prediction, target)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        # Calculate metrics
        pred_classes = torch.argmax(prediction, dim=1).cpu().numpy()
        target_classes = target.cpu().numpy()
        
        losses.append(loss.item())
        accuracies.append(accuracy_score(target_classes, pred_classes))
        f1_scores.append(f1_score(target_classes, pred_classes, average='weighted', zero_division=0))
        precisions.append(precision_score(target_classes, pred_classes, average='weighted', zero_division=0))
        
        progress_bar.set_postfix({
            'loss': f'{np.mean(losses):.4f}',
            'accuracy': f'{np.mean(accuracies):.4f}',
            'f1': f'{np.mean(f1_scores):.4f}',
            'precision': f'{np.mean(precisions):.4f}'
        })
    
    return np.mean(losses), np.mean(accuracies)

def compute_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            pred_classes = torch.argmax(prediction, dim=1).cpu().numpy()
            all_preds.extend(pred_classes)
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('mnist_confusion_matrix.png')
    plt.close()
    
    return cm

def train(model, train_loader, val_loader, loss_fn, optimiser, device, epochs):
    loss_history = []
    acc_history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss, avg_acc = train_single_epoch(model, train_loader, loss_fn, optimiser, device)
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
    
    # Plot and save loss and accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), loss_history, 'b-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), acc_history, 'g-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mnist_training_metrics.png')
    plt.close()
    
    # Compute and save confusion matrix
    compute_confusion_matrix(model, val_loader, device)
    
    return loss_history, acc_history

if __name__ == "__main__":
    # Download data and create data loaders
    train_data, val_data = download_mnist_datasets()
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    val_dataloader = create_data_loader(val_data, BATCH_SIZE)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    
    # Initialize model, loss, and optimizer
    model = NN().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)
    
    # Save model
    torch.save(model.state_dict(), "mnist_example/model/nn.pth")
    print("Trained feed forward net saved at model/nn.pth")