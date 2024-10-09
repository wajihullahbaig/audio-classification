# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 07:28:31 2024

@author: Acer
"""

from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os

class UrbanSoundDataset(Dataset):
    
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device
                 ):
        # Taking annotation from the dataset
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device
        self.transformation = transformation.to(device)
        
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__ (self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal,sr)        
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal,label
    
    def _cut_if_necessary(self,signal):
        # signal -> Tensor -> (1,num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:,:self.num_samples]            
        return signal 
    
    def _right_pad_if_necessary(self,signal):
        # signal -> Tensor -> (1,num_samples)
        if signal.shape[1] < self.num_samples:
            n_samples_to_pad = self.num_samples - signal.shape[1]
            last_dim_padding = (0,n_samples_to_pad)
            signal = torch.nn.functional.pad(signal,last_dim_padding)
        return signal 
    
        
    def _resample_if_necessary(self,signal,sr):
        if sr !=  self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr,self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self,signal):
        # Mix down multiple channels down to one
        if signal.shape[0] > 1:
            signal = torch.mean(signal,dim=0, keepdim = True)
        return signal
        
        
        return signal
        
    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir,fold,self.annotations.iloc[index,0])
        return path
    
    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,6]
    
    

if __name__ == "__main__"    :
    print(str(torchaudio.list_audio_backends()))
    annotations_file = "C:/Users/Acer/Downloads/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir  = "C:/Users/Acer/Downloads/UrbanSound8K/UrbanSound8K/audio"
    sampling_rate = 22050
    num_samples = 22050
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
        
    print(f"Using device: {device}")    
   
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate = sampling_rate, 
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
        )
    
    usd = UrbanSoundDataset(annotations_file,audio_dir,mel_spectrogram_transform,sampling_rate,num_samples,device)
    
    print(len(usd))
    
    
    signal, label = usd[0]