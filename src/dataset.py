from PIL import Image
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from model import ClothesModel
import torch.nn as nn
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ClothesDataset():

    def __init__(self, csv_file, image_dir, label_columns, transform=None):
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.label_columns = label_columns
        self.transform = transform

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_name = row["image"]

        labels = torch.tensor(
            row[self.label_columns].values.argmax(),
            dtype=torch.long
        )

        image_path = self.image_dir + "/" + image_name
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, labels



def load_image(path):
    image = Image.open(path)
    return image

def get_dataset_info(csv_file):
    data = pd.read_csv(csv_file)
    num_images = len(data)
    label_columns = data.columns[1:]

    return num_images, label_columns


if __name__ == "__main__":
    csv_file = "../data/labels.csv"
    num_images, label_columns = get_dataset_info(csv_file)
    dataset = ClothesDataset(csv_file="../data/labels.csv", image_dir="../data/images", label_columns=label_columns, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    print(type(dataloader))

    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)


    model = ClothesModel(num_classes=len(label_columns))
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    images, labels = next(iter(dataloader))

    outputs = model(images)
    loss = loss_fn(outputs, labels)

    print("Before:", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs_after = model(images)
    loss_after = loss_fn(outputs_after, labels)

    print("After:", loss_after.item())