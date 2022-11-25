
from torch.utils.data import DataLoader
from FaceImageDataset import FaceImageDataset

training_data = FaceImageDataset("Dataset/Train/csv/train.csv", "outimg")
train_dataloader = DataLoader(training_data, batch_size=10, shuffle=False)

# Iterate through the DataLoader
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
#plt.imshow(img, cmap="gray")
#plt.show()
print(f"Label: {label}")