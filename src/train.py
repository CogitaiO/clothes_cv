import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ClothesDataset, get_dataset_info, transform
from model import ClothesModel

if __name__ == "__main__":

    csv_file = "../data/labels.csv"

    num_images, label_columns = get_dataset_info(csv_file)

    dataset = ClothesDataset(
        csv_file=csv_file,
        image_dir="../data/images",
        label_columns=label_columns,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    model = ClothesModel(
        num_classes=len(label_columns)
    )

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    for epoch in range(100):

        epoch_loss = 0.0

        for images, labels in dataloader:
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            epoch_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print(loss.item())

        epoch_loss /= len(dataloader)

        print("Epoch loss:", epoch_loss)