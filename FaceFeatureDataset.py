import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from skimage import io, transform
import numpy as np

# https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Dataloader transform 예제


class FaceFeatureDataset(Dataset):
    def __init__(self, feature_file, label_file, feature_indexes = None, label_indexes = None, transform=None, target_transform=None):
        self.landmarks_frame = pd.read_csv(feature_file)
        #self.landmarks_frame = self.landmarks_frame[1:]
        self.labels = pd.read_csv(label_file)
        self.transform = transform
        self.target_transform = target_transform
        self.feature_indexes = feature_indexes
        self.label_indexes = label_indexes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features = self.landmarks_frame.iloc[idx, 1:]       
        features = np.array(features, dtype=np.float32)
        features = features.reshape(-1, 2)
        if self.feature_indexes :
            #print("feature indexes" , self.feature_indexes)            
            features = np.take(features, self.feature_indexes, axis=0)
            #features = features.take(self.features_indexes, axis=0)
            #print(features)
        
        label = self.labels.iloc[idx, 1:]
        label = torch.FloatTensor(label)
        if self.transform :
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label