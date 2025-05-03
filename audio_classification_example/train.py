# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:56:29 2024

@author: Acer
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchaudio
from cnn import CNNNetwork
from urbansound_dataset import UrbanSoundDataset
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import seaborn as sns


BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.0001  # Reduced from 0.001 to stabilize training
VALIDATION_SPLIT = 0.2
MAX_GRAD_NORM = 1.0  # For gradient clipping

def create_data_loader(dataset, batch_size, validation_split=VALIDATION_SPLIT):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    
    return train_loader, val_loader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    losses = []
    accuracies = []
    f1_scores = []
    precisions = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    for input, target in progress_bar:
        input, target = input.to(device), target.to(device)
        
        # Calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("Warning: NaN loss detected. Skipping this batch.")
            continue
        
        # Backpropagate and update weights
        optimiser.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimiser.step()
        
        # Calculate metrics
        pred_classes = torch.argmax(prediction, dim=1).cpu().numpy()
        target_classes = target.cpu().numpy()
        
        losses.append(loss.item())
        accuracies.append(accuracy_score(target_classes, pred_classes))
        f1_scores.append(f1_score(target_classes, pred_classes, average='weighted', zero_division=0))
        precisions.append(precision_score(target_classes, pred_classes, average='weighted', zero_division=0))
        
        progress_bar.set_postfix({
            'loss': f'{np.mean(losses):.4f}' if losses else 'NaN',
            'accuracy': f'{np.mean(accuracies):.4f}',
            'f1': f'{np.mean(f1_scores):.4f}',
            'precision': f'{np.mean(precisions):.4f}'
        })
    
    return np.mean(losses) if losses else float('nan'), np.mean(accuracies)

def compute_confusion_matrix(model, data_loader, device, num_classes=10):
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
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return cm

def train(model, train_loader, val_loader, loss_fn, optimiser, device, epochs, model_path):
    loss_history = []
    acc_history = []
    prev_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss, avg_acc = train_single_epoch(model, train_loader, loss_fn, optimiser, device)
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        
        # Save model if loss improves
        if not np.isnan(avg_loss) and avg_loss < prev_loss:
            prev_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"Trained net saved at {model_path}")
    
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
    plt.savefig('training_metrics.png')
    plt.close()
    
    # Compute and save confusion matrix
    compute_confusion_matrix(model, val_loader, device)
    
    return loss_history, acc_history

if __name__ == "__main__":
    annotations_file = R"C:\Users\Precision\Onus\Data\UrbanSound8K\metadata\UrbanSound8K.csv"
    audio_dir =R"C:\Users\Precision\Onus\Data\UrbanSound8K/audio"
    model_path = os.path.join(os.getcwd(), "audio_classification_example", "model", "cnn.pth")
    sampling_rate = 22050
    num_samples = 22050
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize dataset and data loaders
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = UrbanSoundDataset(annotations_file, audio_dir, mel_spectrogram_transform, sampling_rate, num_samples, device)
    print(f"Dataset size: {len(usd)}")
    
    train_dataloader, val_dataloader = create_data_loader(usd, BATCH_SIZE)
    
    # Initialize model, loss, and optimizer
    model = CNNNetwork().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS, model_path)
    print("Finished training")