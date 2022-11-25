import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

# https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
# Dataloader transform 예제

class FaceImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}' + ".png")
        print('image path ' + img_path)
        image = read_image(img_path)
        #label = self.img_labels.iloc[idx, 1]
        label = torch.Tensor(self.img_labels.iloc[idx, 1:])
        # print(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label