# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:28:53 2024

@author: Acer
"""

import torch
from train import CNNNetwork
from urbansound_dataset import UrbanSoundDataset
import torchaudio
import os

class_mapping = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music"        
        
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
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'        
    print(f"Using device: {device}")    
    
    # load the model
    model = CNNNetwork()
    model_path = os.path.join(os.getcwd(),"audio_classification_example","model","cnn.pth")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    annotations_file = "C:/Users/Acer/Downloads/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir  = "C:/Users/Acer/Downloads/UrbanSound8K/UrbanSound8K/audio"
    sampling_rate = 22050
    num_samples = 22050
        
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate = sampling_rate, 
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
        )
    
    # load the datset
    usd = UrbanSoundDataset(annotations_file,audio_dir,mel_spectrogram_transform,sampling_rate,num_samples,device)
    
    
    # get a sample from validation for inference
    
    input,target = usd[0][0],usd[0][1] # [channels,features,time]
    # CNN needs 4 dimensions, so we unsqueeze
    input.unsqueeze_(1) # [batch_size,channels,features,time]

    #make an inference
    
    predicted, expected = predict(model,input,target, class_mapping)
    
    print(f"Predicted: {predicted}, Expected: {expected}")