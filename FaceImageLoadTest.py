
from torch.utils.data import DataLoader
from FaceImageDataset import FaceImageDataset
import matplotlib.pyplot as plt

training_data = FaceImageDataset(annotations_file="Dataset/Train/csv/train.csv", img_dir="outimg")
train_dataloader = DataLoader(training_data, batch_size=10, shuffle=False)

# fig = plt.figure()

# Iterate through the DataLoader
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# img = train_features[0].squeeze()
img = train_features[0]
label = train_labels[0]
plt.imshow(img)
plt.show()

print(f"Label: {label}")