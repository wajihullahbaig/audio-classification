# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:56:29 2024

@author: Acer
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import torchaudio
from cnn import CNNNetwork
from urbansound_dataset import UrbanSoundDataset
import os

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001




def download_mnist_datasets():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    train_data = datasets.MNIST(
        root= "data",
        download=True,
        train = True,
        transform = transform
        )
    validation_data = datasets.MNIST(
        root= "data",
        download=True,
        train = False,
        transform = transform
        )
    
    return train_data, validation_data


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad() # zero out the gradients or they will accumulate
        loss.backward() # back propagate the gradients
        optimiser.step() # update the weights

    print(f"loss: {loss.item()}")
    
    return loss.item()


def train(model, data_loader, loss_fn, optimiser, device, epochs,model_path):
    prev_loss = 0.0
    for i in range(epochs):
        print(f"Epoch {i+1}")
        loss = train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
        if  prev_loss > loss or prev_loss == 0.0:
            # save model which has a better loss than the previous one
            prev_loss = loss
            
            torch.save(model.state_dict(), model_path)
            print(f"Trained net saved at {model_path}")    


if __name__ == "__main__":
    
    annotations_file = "C:/Users/Acer/Downloads/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir  = "C:/Users/Acer/Downloads/UrbanSound8K/UrbanSound8K/audio"
    model_path = os.path.join(os.getcwd(),"audio_classification_example","model","cnn.pth")
    sampling_rate = 22050
    num_samples = 22050
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'        
    print(f"Using device: {device}")    
    
    # instantiate out dataset and and create data loader

   
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate = sampling_rate, 
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
        )
    usd = UrbanSoundDataset(annotations_file,audio_dir,mel_spectrogram_transform,sampling_rate,num_samples,device)
    
    print(len(usd))
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

   
    model = CNNNetwork().to(device)
    print(model)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(model, train_dataloader, loss_fn, optimiser, device, EPOCHS,model_path)
    print("Finished training")


