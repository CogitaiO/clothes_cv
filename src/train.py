import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ClothesDataset, get_dataset_info, transform
from model import ClothesModel

if __name__ == "__main__":
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"

    _, label_columns = get_dataset_info(train_csv)

    train_dataset = ClothesDataset(
        csv_file=train_csv,
        image_dir="data/images",
        label_columns=label_columns,
        transform=transform
    )

    val_dataset = ClothesDataset(
        csv_file=val_csv,
        image_dir="data/images",
        label_columns=label_columns,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
    )

    model = ClothesModel(
        num_classes=len(label_columns),
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    for epoch in range(20):
        model.train()

        model.backbone.eval()
        model.backbone.fc.train()

        train_loss = 0

        for images, labels in train_loader:
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)

                loss = loss_fn(outputs, labels)

                val_loss += loss.item()

            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )
        model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)

                predictions = probabilities.argmax(dim=1)

                print("True:", labels)
                print("Pred:", predictions)
                print("Probabilities:", probabilities)
    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels)
    print(labels.shape)
    print(labels.dtype)