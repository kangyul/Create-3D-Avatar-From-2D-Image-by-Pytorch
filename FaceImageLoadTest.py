
from torch.utils.data import DataLoader
from FaceImageDataset import FaceImageDataset
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets, models

'''
training_data = FaceImageDataset(annotations_file="Dataset/Train/csv/train.csv", img_dir="outimg/Train", transform=transforms.Compose([
            transforms.ToTensor(), # tensor 타입으로 데이터 변경
            transforms.Normalize(mean = (0.5,), std = (0.5,)) # data를 normalize 하기 위한 mean과 std 입력
        ]))
'''

training_data = FaceImageDataset(annotations_file="Dataset/Train/csv/train.csv", img_dir="outimg/Train")

train_dataloader = DataLoader(training_data, batch_size=10, shuffle=False)

# fig = plt.figure()

# Iterate through the DataLoader
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

#img = train_features[0].squeeze()
img = train_features[0]
label = train_labels[0]

print(img)

# plt.imshow(img)
# plt.show()
print(f"Label: {label}")