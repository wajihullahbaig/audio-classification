# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:28:53 2024

@author: Acer
"""

import torch
from train import NN, download_mnist_datasets

class_mapping = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",        
        
    ]

def predict(model,input,target,class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor  -> (batch_size,class)        
        predicted_index = predictions[0].argmax(0) # Index of max class prob
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected
    

if __name__ == "__main__":
    # load the model
    model = NN()
    state_dict = torch.load("model/nn.pth")
    model.load_state_dict(state_dict)

    # load the dataset
    
    _, validation_dataset = download_mnist_datasets()
    
    # get a sample from validation for inference
    
    input,target = validation_dataset[0][0],validation_dataset[0][1]

    #make an inference
    
    predicted, expected = predict(model,input,target, class_mapping)
    
    print(f"Predicted: {predicted}, Expected: {expected}")