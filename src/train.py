import os
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
        transform=transform,
    )

    val_dataset = ClothesDataset(
        csv_file=val_csv,
        image_dir="data/images",
        label_columns=label_columns,
        transform=transform,
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

    # Гарантируем, что градиенты считаются только для классификатора
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()

    # Передаем в оптимизатор только обучаемые параметры
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
    )

    best_val_loss = float("inf")
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(20):
        model.train()
        model.backbone.eval()
        model.backbone.fc.train()

        train_loss = 0.0

        for images, labels in train_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        with open("logs.txt", "w", encoding="utf-8") as f:
            f.write()
        print(
            f"Epoch {epoch + 1:02d}/20: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"best_val_loss={best_val_loss:.4f}"
        )