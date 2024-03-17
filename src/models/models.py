import torch
import os
import pickle
from torch import nn
from torch.utils.data import DataLoader
from data import dataset


class NeuralModel:
    """
    Wrapper for neural network models used in this project.
    """

    def __init__(self, model: nn.Module, model_file: str):
        """
        Init function
        Params:
          - model: The model to wrap
          - model_file: The file the model should be stored in, if applicable
        """
        self.model = model
        self.model_file = model_file
        if model_file == "":
            raise ValueError("Model file must be provided")

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        max_epochs: int = 50,
        early_stop: int = 5,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        save_loss=True,
    ):
        """
        Train the contained model
        Params:
          - train_loader: Data loader for the training dataset
          - valid_loader: Data loader for the validation dataset
          - max_epochs: Maximum number of epochs before stopping training
          - early_stop: If validation loss increases for this many epochs, stop training
          - loss_fn: The loss function to optimize
          - optimizer: The optimizer to use on the model
        """
        losses = self.load_loss()
        if len(losses) == 0:
            best_valid_loss = self.eval(valid_loader, loss_fn)
            best_epoch = 0
            losses.append(best_valid_loss)
        else:
            best_valid_loss = max(losses)
            best_epoch = losses.index(best_valid_loss)
        stop_count = 0
        optimizer = optimizer(self.model.parameters())
        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}:")
            size = len(train_loader.dataset)
            current = 0
            self.model.train()
            for batch, (X, y, _) in enumerate(train_loader):
                # Compute prediction error
                pred = self.model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                current += len(X)
                if batch % 100 == 0:  # Print progress every 100 batches
                    loss = loss.item()
                    print(f"  loss: {loss:.5f}\t[{current-len(X)}-{current}/{size}]")
            valid_loss = self.eval(valid_loader)
            losses.append(valid_loss)
            print(f"  Validation Loss: {valid_loss} (previous best {best_valid_loss} on epoch {best_epoch})")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch + 1
                stop_count = 0
                torch.save(self.model, self.model_file)
                print(f"  Saving updated model to {self.model_file}")
                if save_loss:  # only save updated losses if the model itself is being saved
                    save_loss(losses)
            else:
                stop_count += 1
                if stop_count >= early_stop:
                    print(f"  Exiting model training early after {epoch + 1} epochs")
                    return

    def save_loss(self, loss):
        with open(f"{self.model_file}.loss", "wb") as f:
            pickle.dump(loss, f)

    def load_loss(self):
        try:
            with open(f"{self.model_file}.loss", "rb") as f:
                return pickle.load(f)
        except Exception:
            return []

    def eval(self, test_loader: DataLoader, loss_fn=nn.MSELoss()):
        """
        Evaluate the model on a dataset
        Params:
          - test_loader: Data loader for dataset to evaluate on
          - loss_fn: The loss function to evaluate
        Returns: (loss on preprocessed data, loss on raw data)
        """
        self.model.eval()
        preproc = test_loader.dataset.preprocessor
        total_loss = 0
        total_loss_raw = 0
        size = len(test_loader.dataset)
        with torch.no_grad():
            for _, (X, y, unscale_params) in enumerate(test_loader):
                # Compute prediction error on preprocessed data
                pred = self.model(X)
                total_loss += loss_fn(pred, y).item() * len(X)

                # Compute prediction error on raw data
                total_loss_raw += loss_fn(
                    preproc.unscaleLabels(pred, unscale_params), preproc.unscaleLabels(y, unscale_params)
                ).item() * len(X)
        return (total_loss / size, total_loss_raw / size)


class BasicFeedForward(nn.Module):
    """
    Basic feedforward neural network for stock price prediction
    """

    def __init__(self):
        """
        Initialize the neural network layers
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dataset.INPUT_DAYS * 5, dataset.INPUT_DAYS),
            nn.Tanh(),
            nn.Linear(dataset.INPUT_DAYS, dataset.OUTPUT_DAYS),
            nn.ReLU(),
            nn.Linear(dataset.OUTPUT_DAYS, dataset.OUTPUT_DAYS),
        )

    def forward(self, x):
        """
        Run a forward pass of the network
        Params:
          - x: The input variables
        """
        return self.linear_relu_stack(self.flatten(x))


def load_model(model_name: str, model_file: str):
    """
    Get a model wrapper by the name of the model and the file it should be stored at
    Params:
      - model_name: The name of the model
      - model_file: Path to file where model is stored
    Returns: A wrapper around the model
    """
    if model_name is None and model_file is None:
        raise ValueError("Either model type or model file must be provided")
    if os.path.exists(model_file):
        return NeuralModel(torch.load(model_file), model_file)
    match model_name:
        case "BasicFeedForward":
            return NeuralModel(BasicFeedForward(), model_file)
        case _:
            raise ValueError(f"No model by name '{model_name}' and no file at '{model_file}'")
