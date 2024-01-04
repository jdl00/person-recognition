import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvBlock, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stateless
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class MultiOutputConv(nn.Module):
    def __init__(self) -> None:
        super(MultiOutputConv, self).__init__()

        # Conv blocks
        self.block1 = ConvBlock(in_channels=1, out_channels=64)
        self.block2 = ConvBlock(in_channels=64, out_channels=128)
        self.block3 = ConvBlock(in_channels=128, out_channels=256)
        self.block4 = ConvBlock(in_channels=256, out_channels=1024)

        # Flatten
        self.flatten = nn.Flatten()

        # Set the flattened size
        flattened_size = 1024 * 3 * 3

        # Fully connected layers - shared
        self.fc_shared = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # 6 age classes
        self.fc_age = nn.Linear(512, 6)
        # 8 ethnicity classes
        self.fc_ethnicity = nn.Linear(512, 4)
        # 2 Gender classes
        self.fc_gender = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten and fully connected layer
        x = self.flatten(x)

        x = self.fc_shared(x)

        # Output predictions for each task
        age = self.fc_age(x)
        ethnicity = self.fc_ethnicity(x)
        gender = self.fc_gender(x)

        return age, ethnicity, gender
