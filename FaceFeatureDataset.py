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
    def __init__(self, feature_file, label_file, transform=None, target_transform=None):
        self.landmarks_frame = pd.read_csv(feature_file)
        self.labels = pd.read_csv(label_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype(np.float32).reshape(-1, 2)        
        label = torch.Tensor(self.labels.iloc[idx, 1:]) #, dtype=torch.float64)        
        if self.transform :
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return landmarks, label
