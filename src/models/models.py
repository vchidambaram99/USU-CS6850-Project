import torch
import os
import pickle
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression


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
        training_losses, valid_losses = self.load_losses()
        if len(training_losses) == 0:
            best_valid_loss = self.eval(valid_loader, loss_fn)
            best_epoch = 0
            valid_losses.append(best_valid_loss)
        else:
            best_valid_loss = min(valid_losses)
            best_epoch = valid_losses.index(best_valid_loss)  # note: 0 here is before training

            # Clear old data not saved to the model
            training_losses = training_losses[:best_epoch]
            valid_losses = valid_losses[: best_epoch + 1]

        optimizer = optimizer(self.model.parameters())
        for epoch in range(best_epoch, max_epochs):
            print(f"Epoch {epoch + 1}:")
            train_loader.dataset.shuffle()
            size = len(train_loader.dataset)
            current = 0
            train_loss = 0
            self.model.train()
            for batch, (X, y, _) in enumerate(train_loader):
                # Compute prediction error
                pred = self.model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item() * len(X)

                # Print progress every 100 batches
                current += len(X)
                if batch % 100 == 0:
                    loss = loss.item()
                    print(f"  loss: {loss:.5f}\t[{current-len(X)}-{current}/{size}]")

            training_losses.append(train_loss / len(train_loader.dataset))
            print(f"  Training loss: {training_losses[-1]}")

            # Evaluate on validation set to determine if training should be stopped early
            valid_loader.dataset.shuffle()
            valid_loss = self.eval(valid_loader)
            valid_losses.append(valid_loss)
            print(f"  Validation Loss: {valid_loss} (previous best {best_valid_loss} on epoch {best_epoch})")

            if save_loss:
                self.save_losses(training_losses, valid_losses)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.model_file)
                print(f"  Saving updated model to {self.model_file}")
            elif epoch - best_epoch + 1 >= early_stop:
                print(f"  Exiting model training early after {epoch + 1} epochs")
                return

    def save_losses(self, train_losses, valid_losses):
        with open(f"{self.model_file}.loss", "wb") as f:
            pickle.dump({"train": train_losses, "valid": valid_losses}, f)

    def load_losses(self):
        try:
            with open(f"{self.model_file}.loss", "rb") as f:
                losses = pickle.load(f)
                return losses["train"], losses["valid"]
        except Exception:
            return [], []

    def eval(self, test_loader: DataLoader, loss_fn=nn.MSELoss()):
        """
        Evaluate the model on a dataset
        Params:
          - test_loader: Data loader for dataset to evaluate on
          - loss_fn: The loss function to evaluate
        Returns: Average loss
        """
        self.model.eval()
        total_loss = 0
        size = len(test_loader.dataset)
        with torch.no_grad():
            for _, (X, y, unscale_params) in enumerate(test_loader):
                # Compute prediction error on preprocessed data
                pred = self.model(X)
                total_loss += loss_fn(pred, y).item() * len(X)
        return total_loss / size


class PermuteLayer(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return torch.permute(x, self.permutation)


class ConfigurableNetwork(nn.Module):
    """
    NeuralNetwork configurable by JSON file
    """

    def __init__(self, model_config: dict):
        """
        Initialize the neural network layers
        Params:
          - model_config: Dictionary describing model structure
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in model_config["layers"]:
            if layer["type"] == "PermuteLayer":
                LayerClass = PermuteLayer
            else:
                LayerClass = getattr(nn, layer["type"])
            self.layers.append(LayerClass(*layer.get("args", []), **layer.get("kwargs", {})))

    def forward(self, x):
        """
        Run a forward pass of the network
        Params:
          - x: The input variables
        """
        for layer in self.layers:
            if type(layer) is nn.LSTM:  # LSTM returns (output, hidden state)
                x = layer(x)[0]
            else:
                x = layer(x)
        return x


class LinearExtrapolationModel:
    """
    Linear extrapolation baseline model
    Note that this model only works if the preprocessed inputs can be extrapolated directly to the outputs
    """

    def __init__(self):
        """
        Init function
        """
        pass

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ):
        """
        Train the model. Since this model requires no training, just evaluate it on both sets
        Params:
          - train_loader: Data loader for the training dataset
          - valid_loader: Data loader for the validation dataset
        """
        print(f"Loss on training set: {self.eval(train_loader)}")
        print(f"Loss on validation set: {self.eval(valid_loader)}")

    def eval(self, test_loader: DataLoader, loss_fn=nn.MSELoss()):
        """
        Evaluate the model on a dataset
        Params:
          - test_loader: Data loader for dataset to evaluate on
          - loss_fn: The loss function to evaluate
        Returns: Average loss
        """
        total_loss = 0
        size = len(test_loader.dataset)
        with torch.no_grad():
            for _, (X, y, unscale_params) in enumerate(test_loader):
                # Compute prediction error on preprocessed data
                pred = torch.zeros(y.shape)
                for i in range(len(X)):
                    # Build model for extrapolation
                    model = LinearRegression()
                    xs = np.arange(len(X[i][0])).reshape(-1, 1)
                    ys = X[i][0].numpy()
                    model.fit(xs, ys)

                    # Extrapolate for predictions
                    xs = (np.arange(len(y[i])) + len(X[i][0])).reshape(-1, 1)
                    predictions = model.predict(xs)
                    pred[i] = torch.Tensor(predictions)

                # Calculate loss
                total_loss += loss_fn(pred, y).item() * len(X)
        return total_loss / size


def load_model(model_config: dict, model_file: str):
    """
    Get a model wrapper by configuration and optionally the file it should be stored at
    Params:
      - model_config: The configuration of the model
      - model_file: Path to file where model is stored
    Returns: A wrapper around the model
    """
    model_type = model_config["type"]
    match model_type:
        case "NeuralNetwork":
            network = ConfigurableNetwork(model_config)
            if os.path.exists(model_file):
                network.load_state_dict(torch.load(model_file))
            return NeuralModel(network, model_file)
        case "LinearExtrapolation":
            return LinearExtrapolationModel()
        case _:
            raise ValueError(f"No model of type '{model_type}'")
