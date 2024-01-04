import os
import torch

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from enum import Enum


DATASET_NAME = os.path.join(os.path.abspath("datasets"), "cleaned_dataset.csv")


class AgeEncoder(Enum):
    FIRST = (0, "0-18")
    SECOND = (1, "18-35")
    THIRD = (2, "35-50")
    FOURTH = (3, "50-65")
    FIFTH = (4, "65-90")
    SIXTH = (5, "90+")

    @staticmethod
    def encode(age_string):
        return AgeEncoder._map.get(age_string, None).value[0]


# Now, to create the _map dictionary
AgeEncoder._map = {
    age_range: enum_member
    for enum_member in AgeEncoder
    for _, age_range in [enum_member.value]
}


def extract_csv():
    data = pd.read_csv(DATASET_NAME)
    # Preprocess the data
    # Convert pixels into numpy array
    data["pixels"] = data["pixels"].apply(
        lambda x: np.array(x.split(), dtype=np.float32)
    )
    X = np.stack(data["pixels"].values) / 255.0
    y_age = data["age"].apply(lambda age: AgeEncoder.encode(age)).values
    y_ethnicity = data["ethnicity"].values
    y_gender = data["gender"].values
    return X, y_age, y_ethnicity, y_gender


def transform_x(X, y_age, y_ethnicity, y_gender):
    X_reshaped = X.reshape((-1, 48, 48))

    (
        X_train,
        _,
        y_age_trans,
        _,
        y_ethnicity_trans,
        _,
        y_gender_trans,
        _,
    ) = train_test_split(
        X_reshaped, y_age, y_ethnicity, y_gender, test_size=0.2, random_state=42
    )

    # Image transformations
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.RandomCrop(48),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    X_trans = np.array([transform(image).numpy() for image in X_train])

    return (
        torch.tensor(X_trans).float(),
        torch.tensor(y_age_trans).long(),
        torch.tensor(y_ethnicity_trans).long(),
        torch.tensor(y_gender_trans).long(),
    )


def get_datasets():
    X, y_age, y_ethnicity, y_gender = extract_csv()
    X_reshaped = X.reshape((-1, 1, 48, 48))

    (
        X_train,
        X_val,
        y_age_train,
        y_age_val,
        y_ethnicity_train,
        y_ethnicity_val,
        y_gender_train,
        y_gender_val,
    ) = train_test_split(
        X_reshaped, y_age, y_ethnicity, y_gender, test_size=0.2, random_state=42
    )

    (X_trans, y_age_trans, y_ethnicity_trans, y_gender_trans) = transform_x(
        X, y_age, y_ethnicity, y_gender
    )

    # Convert train to PyTorch tensors
    X_train_tensor = torch.tensor(X_train).float()
    y_age_train_tensor = torch.tensor(y_age_train).long()
    y_ethnicity_train_tensor = torch.tensor(y_ethnicity_train).long()
    y_gender_train_tensor = torch.tensor(y_gender_train).long()

    # Concat the normal with transformed tensors
    X_train_tensor = torch.cat((X_train_tensor, X_trans), dim=0)
    y_age_train_tensor = torch.cat((y_age_train_tensor, y_age_trans), dim=0)
    y_ethnicity_train_tensor = torch.cat(
        (y_ethnicity_train_tensor, y_ethnicity_trans), dim=0
    )
    y_gender_train_tensor = torch.cat((y_gender_train_tensor, y_gender_trans), dim=0)

    # Convert val to PyTorch tensors
    X_val_tensor = torch.tensor(X_val).float()
    y_age_val_tensor = torch.tensor(y_age_val).long()
    y_ethnicity_val_tensor = torch.tensor(y_ethnicity_val).long()
    y_gender_val_tensor = torch.tensor(y_gender_val).long()

    # Create Data Loaders
    train_dataset = TensorDataset(
        X_train_tensor,
        y_age_train_tensor,
        y_ethnicity_train_tensor,
        y_gender_train_tensor,
    )
    # Create Data Loaders
    val_dataset = TensorDataset(
        X_val_tensor,
        y_age_val_tensor,
        y_ethnicity_val_tensor,
        y_gender_val_tensor,
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader
