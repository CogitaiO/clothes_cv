import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/labels.csv")

label_columns = data.columns[1:]

classes = data[label_columns].idxmax(axis=1)

train_data, val_data = train_test_split(
    data,
    test_size=0.25,
    random_state=42,
    stratify=classes
)

train_data.to_csv("data/train.csv", index=False)
val_data.to_csv("data/val.csv", index=False)

print(train_data[label_columns].sum())
print(val_data[label_columns].sum())