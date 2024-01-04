import os

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MultiOutputConv
from dataset import get_datasets

CHECKPOINT_FOLDER = os.path.abspath("checkpoints")
OUTPUT_FOLDER = os.path.abspath("output")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim,
    val_loss: float,
):
    file_path = os.path.join(CHECKPOINT_FOLDER, f"checkpoint_{epoch+1}.pth")

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(state, file_path)


class Metrics:
    @staticmethod
    def calculate_accuracy(y_pred_tuple, y_true_tuple):
        """
        Calculate the accuracy for each output of a multi-head model.
        Returns a tuple of accuracies for each task.

        Parameters:
        - y_pred_tuple: A tuple of Tensors of predicted outputs (age_pred, ethnicity_pred, gender_pred)
        - y_true_tuple: A tuple of Tensors of actual labels (y_age_true, y_ethnicity_true, y_gender_true)

        Returns:
        - A tuple containing accuracies of each task (age_accuracy, ethnicity_accuracy, gender_accuracy)
        """

        # Unpack the tuples
        age_pred, ethnicity_pred, gender_pred = y_pred_tuple
        y_age_true, y_ethnicity_true, y_gender_true = y_true_tuple

        # Calculate accuracies
        age_accuracy = (age_pred.argmax(1) == y_age_true).float().mean()
        ethnicity_accuracy = (
            (ethnicity_pred.argmax(1) == y_ethnicity_true).float().mean()
        )
        gender_accuracy = (gender_pred.argmax(1) == y_gender_true).float().mean()

        return age_accuracy.item(), ethnicity_accuracy.item(), gender_accuracy.item()


class Regularization:
    # min delta dunno what for?
    _min_delta = 0.01

    # Best validation loss
    _best_val_loss = float("inf")

    # Patience for counter
    _patience_counter = 0

    def __init__(self, patience: int = 8) -> None:
        self._es_patience = patience

    def early_stop(
        self,
        c_epoch: int,
        val_loss: float,
        model: nn.Module,
    ) -> bool:
        """Stops training if the validation loss changes

        Args:
            c_epoch (int): The current epoch.
            model (nn.Module): The model being trained.

        Returns:
            bool: Whether training should stop
        """

        print((f"Patience Counter: {self._patience_counter}/{self._es_patience}"))
        print(f"Best Validation Loss: {self._best_val_loss}\n")

        # Check for early stopping
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            output_path = os.path.join(
                OUTPUT_FOLDER, f"best_val_loss_model_{c_epoch}.pth"
            )
            torch.save(model.state_dict(), output_path)

            return False
        else:
            # Validation loss has not improved increment counter
            self._patience_counter += 1
            if self._patience_counter >= self._es_patience:
                output_path = os.path.join(OUTPUT_FOLDER, f"es_model_{c_epoch}.pth")
                torch.save(model.state_dict(), output_path)

                print("Early stopping triggered.")
                return True

            return False


class TrainerOutput:
    @staticmethod
    def pre_epoch(epoch: int):
        print(f"Training Epoch: {epoch+1}")

    @staticmethod
    def post_train(train_loss: float, train_loader: DataLoader, train_accuracy: float):
        print("\nCompleted Training:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(
            f"Train Accuracy: Age - {train_accuracy[0]:.2f}, Ethnicity - {train_accuracy[1]:.2f}, Gender - {train_accuracy[2]:.2f}"
        )
        print("")

    @staticmethod
    def post_val(val_loss: float, val_loader: DataLoader, val_accuracy: list):
        print("Completed Validation:")
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        print(
            f"Validation Accuracy: Age - {val_accuracy[0]:.2f}, Ethnicity - {val_accuracy[1]:.2f}, Gender - {val_accuracy[2]:.2f}"
        )


# TODO: Implement checkpoint restore
class Trainer:
    """Trainer object for training the model"""

    # Whether the model should stop training
    _exit = False

    # The criterion to be used
    _criterion = nn.CrossEntropyLoss()
    # Device for model/datasets to be set to
    _device = torch.device(get_device())
    # Epochs of training
    _n_epochs = 50

    # Train metrics
    _train_loss = 0
    _train_accuracy = []

    # Validation metrics
    _val_loss = 0
    _val_accuracy = []
    _val_loss_per_batch = 0

    def __init__(
        self,
        train_loader,
        val_loader,
        model: nn.Module,
        checkpoint: Optional[str] = None,
    ) -> None:
        self._optimizer = optim.Adam(model.parameters(), lr=0.001)

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._checkpoint = checkpoint
        self._regularlisation = Regularization()

        self._scheduler = lr_scheduler.ExponentialLR(self._optimizer, gamma=0.7)

    def _callbacks(self, c_epoch: int):
        """Callbacks to run after training."""
        self._exit = self._regularlisation.early_stop(
            c_epoch=c_epoch, val_loss=self._val_loss_per_batch, model=self._model
        )

    def _train_epoch(self, epoch: int) -> None:
        # Lets use tqdm to get some cool training bars
        with tqdm(
            total=len(self._train_loader),
            desc=f"Epoch {epoch + 1}/{self._n_epochs}",
            unit="batch",
        ) as pbar:
            for (
                X_batch,
                y_age_batch,
                y_ethnicity_batch,
                y_gender_batch,
            ) in self._train_loader:
                # Move the batches to the device
                X_batch = X_batch.to(self._device)
                y_age_batch = y_age_batch.to(self._device)
                y_ethnicity_batch = y_ethnicity_batch.to(self._device)
                y_gender_batch = y_gender_batch.to(self._device)

                # Forward pass
                age_pred, ethnicity_pred, gender_pred = self._model(X_batch)

                # Calculate loss
                loss = (
                    self._criterion(age_pred, y_age_batch)
                    + self._criterion(ethnicity_pred, y_ethnicity_batch)
                    + self._criterion(gender_pred, y_gender_batch)
                )

                # Backward pass and optimize
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                self._train_loss += loss.item()
                self._train_accuracy.append(
                    Metrics.calculate_accuracy(
                        (age_pred, ethnicity_pred, gender_pred),
                        (y_age_batch, y_ethnicity_batch, y_gender_batch),
                    )
                )

                pbar.update(1)
                pbar.set_postfix_str(
                    f"Training Loss: {self._train_loss / (pbar.n + 1):.4f}"
                )

        # Calculate mean training accuracy over each output
        train_accuracy = torch.mean(torch.tensor(self._train_accuracy), dim=0)
        TrainerOutput.post_train(
            train_loss=self._train_loss,
            train_loader=self._train_loader,
            train_accuracy=train_accuracy,
        )

    def _validate_epoch(self) -> None:
        # Validation step
        self._model.eval()

        with torch.no_grad():
            for (
                X_batch,
                y_age_batch,
                y_ethnicity_batch,
                y_gender_batch,
            ) in tqdm(self._val_loader, desc="Validating", leave=False):
                # Move tensors to the device
                X_batch = X_batch.to(self._device)
                y_age_batch = y_age_batch.to(self._device)
                y_ethnicity_batch = y_ethnicity_batch.to(self._device)
                y_gender_batch = y_gender_batch.to(self._device)

                # Forward pass
                age_pred, ethnicity_pred, gender_pred = self._model(X_batch)

                # Calculate loss
                loss = (
                    self._criterion(age_pred, y_age_batch)
                    + self._criterion(ethnicity_pred, y_ethnicity_batch)
                    + self._criterion(gender_pred, y_gender_batch)
                )

                self._val_loss += loss.item()
                self._val_accuracy.append(
                    Metrics.calculate_accuracy(
                        (age_pred, ethnicity_pred, gender_pred),
                        (y_age_batch, y_ethnicity_batch, y_gender_batch),
                    )
                )

        # Calculate mean validation accuracy over each output
        val_accuracy = torch.mean(torch.tensor(self._val_accuracy), dim=0)

        # Calculate the loss per batch
        self._val_loss_per_batch = self._val_loss / len(self._val_loader)

        # Print the post train output
        TrainerOutput.post_val(
            val_loss=self._val_loss,
            val_loader=self._val_loader,
            val_accuracy=val_accuracy,
        )

    def train(self) -> None:
        self._model = self._model.to(self._device)

        for epoch in range(self._n_epochs):
            self._train_loss = 0
            self._train_accuracy = []
            self._val_loss = 0
            self._val_loss_per_batch = 0
            self._val_accuracy = []

            # Stop the model training
            if self._exit:
                break

            # Pre Epoch print useful training things
            TrainerOutput.pre_epoch(epoch=epoch)

            # Set the model to train
            self._model.train()
            self._train_epoch(epoch=epoch)

            # Null layers and perform validation
            self._model.eval()
            self._validate_epoch()

            # Run the callbacks
            self._callbacks(c_epoch=epoch)

            # Step the scheduler
            self._scheduler.step()

            # Save a checkpoint of training
            save_checkpoint(
                epoch=epoch,
                model=self._model,
                optimizer=self._optimizer,
                val_loss=self._val_loss,
            )


def main():
    train_loader, val_loader = get_datasets()
    model = MultiOutputConv()

    trainer = Trainer(train_loader=train_loader, val_loader=val_loader, model=model)

    trainer.train()


if __name__ == "__main__":
    main()
