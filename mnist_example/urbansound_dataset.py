# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 07:28:31 2024

@author: Acer
"""

from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class UrbanSoundDataset(Dataset):
    
    def __init__(self,annotations_file,audio_dir):
        # Taking annotation from the dataset
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__ (self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,sr = torchaudio.load(audio_sample_path)
        return signal,label
    
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
    usd = UrbanSoundDataset(annotations_file,audio_dir)
    
    print(len(usd))
    
    
    signal, label = usd[0]