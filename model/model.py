import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.dropout = nn.Dropout2d(p=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SharedFC(nn.Module):
    def __init__(self, input_size):
        super(SharedFC, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(x)


class FCOutput(nn.Module):
    def __init__(self, input, classes):
        super(FCOutput, self).__init__()

        self.fc_class = nn.Sequential(
            nn.Linear(input, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_class(x)


class MultiOutputConv(nn.Module):
    def __init__(self) -> None:
        super(MultiOutputConv, self).__init__()

        # Conv blocks
        self.block1 = ConvBlock(in_channels=1, out_channels=16)
        self.block2 = ConvBlock(in_channels=16, out_channels=32)
        self.block3 = ConvBlock(in_channels=32, out_channels=64)
        self.block4 = ConvBlock(in_channels=64, out_channels=256)

        # Flatten
        self.flatten = nn.Flatten()

        # Set the flattened size
        flattened_size = 256 * 3 * 3

        # Fully connected layers - shared
        self.fc_shared = SharedFC(flattened_size)

        # 6 age classes
        self.fc_age = FCOutput(256, 6)
        # 4 ethnicity classes
        self.fc_ethnicity = FCOutput(256, 5)
        # 2 Gender classes
        self.fc_gender = FCOutput(256, 2)

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
        age_logits = self.fc_age(x)
        ethnicity_logits = self.fc_ethnicity(x)
        gender_logits = self.fc_gender(x)
        return age_logits, ethnicity_logits, gender_logits

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass for inference with softmax
        age_logits, ethnicity_logits, gender_logits = self.forward(x)

        # Apply the softmax function
        age_probs = F.softmax(age_logits, dim=1)
        ethnicity_probs = F.softmax(ethnicity_logits, dim=1)
        gender_probs = F.softmax(gender_logits, dim=1)
        return age_probs, ethnicity_probs, gender_probs
