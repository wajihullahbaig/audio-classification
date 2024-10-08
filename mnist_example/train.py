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



BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


class NN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            # Squish the input
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            # Squish till we get number of classes
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions


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


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    # download data and create data loader
    train_data, _ = download_mnist_datasets()
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    model = NN().to(device)
    print(model)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(model, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(model.state_dict(), "model/nn.pth")
    print("Trained feed forward net saved at model/nn.pth")