import os

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
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
    file_path = os.path.join(CHECKPOINT_FOLDER, f"checkpoint_{epoch}.pth")

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
    @staticmethod
    def early_stop(
        c_epoch: int,
        val_loss: float,
        patience_counter: int,
        es_patience: int,
        best_val_loss: float,
        model: nn.Module,
    ) -> tuple:
        """Stops training if the validation loss changes

        Args:
            c_epoch (int): The current epoch.
            val_loss (float): The loss from the validation set.
            patience_counter (int): The amount of times, validation loss decrease.
            es_patience (int): The limit to the early stopping.
            best_val_loss (float): The best validation loss performance.
            model (nn.Module): The model being trained.

        Returns:
            tuple: The current patience counter and whether to continue.
        """
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = os.path.join(OUTPUT_FOLDER, f"best_model_{c_epoch}.pth")

            torch.save(model.state_dict(), output_path)
            save_checkpoint()

            return (patience_counter, True)
        else:
            # Validation loss has not improved increment counter
            patience_counter += 1
            if patience_counter >= es_patience:
                print("Early stopping triggered.")
                return (0, False)

            return (patience_counter, True)


class TrainerOutput:
    @staticmethod
    def pre_epoch(
        epoch: int, patience_counter: int, es_patience: int, best_val_loss: float
    ):
        print(f"Training Epoch: {epoch}")
        print((f"Patience Counter: {patience_counter}/{es_patience}"))
        print(f"Best Validation Loss: {best_val_loss}")

    @staticmethod
    def post_val(val_loss: float, val_loader: DataLoader, val_accuracy):
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        print(
            f"Validation Accuracy: Age - {val_accuracy[0]:.2f}, Ethnicity - {val_accuracy[1]:.2f}, Gender - {val_accuracy[2]:.2f}"
        )


class Trainer:
    """Trainer object for training the model"""

    # Whether the model should stop training
    _exit = False

    # early stopping patience
    _es_patience = 5

    # min delta dunno what for?
    _min_delta = 0.01

    # Best validation loss
    _best_val_loss = float("inf")

    # Patience for counter
    _patience_counter = 0

    # The criterion to be used
    _criterion = nn.CrossEntropyLoss()
    # Device for model/datasets to be set to
    _device = torch.device(get_device())

    # Epochs of training
    _n_epochs = 30

    def __init__(
        self,
        train_loader,
        val_loader,
        model: nn.Module,
        checkpoint: Optional[str] = None,
    ) -> None:
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._checkpoint = checkpoint
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def _callbacks(self):
        """Callbacks to run after training."""
        self._patience_counter, self._exit = Regularization.early_stop()

    def _train_epoch(self, epoch: int):
        train_loss = 0
        train_accuracy = []

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

                train_loss += loss.item()
                train_accuracy.append(
                    Metrics.calculate_accuracy(
                        (age_pred, ethnicity_pred, gender_pred),
                        (y_age_batch, y_ethnicity_batch, y_gender_batch),
                    )
                )

                pbar.update(1)
                pbar.set_postfix_str(f"Training Loss: {train_loss / (pbar.n + 1):.4f}")

        # Calculate mean training accuracy over each output
        train_accuracy = torch.mean(torch.tensor(train_accuracy), dim=0)

    def _validate_epoch(self):
        # Validation step
        self._model.eval()
        val_loss = 0
        val_accuracy = []

        with torch.no_grad():
            for (
                X_batch,
                y_age_batch,
                y_ethnicity_batch,
                y_gender_batch,
            ) in self._val_loader:
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

                val_loss += loss.item()
                val_accuracy.append(
                    Metrics.calculate_accuracy(
                        (age_pred, ethnicity_pred, gender_pred),
                        (y_age_batch, y_ethnicity_batch, y_gender_batch),
                    )
                )

        # Calculate mean validation accuracy over each output
        val_accuracy = torch.mean(torch.tensor(val_accuracy), dim=0)

        # Print the post train output
        TrainerOutput.post_val(
            val_loss=val_loss, val_loader=self._val_loader, val_accuracy=val_accuracy
        )

    def train(self):
        self._model = self._model.to(self._device)

        for epoch in range(self._n_epochs):
            # Stop the model training
            if self._exit:
                break

            # Pre Epoch print useful training things
            self._pre_epoch()

            # Set the model to train
            self._model.train()
            self._train_epoch(epoch=epoch)

            # Null layers and perform validation
            self._model.eval()
            self._validate_epoch()

            # Run the callbacks
            self._callbacks()


def main():
    train_loader, val_loader = get_datasets()
    model = MultiOutputConv()


if __name__ == "__main__":
    main()
