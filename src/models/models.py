import torch
import os
import pickle
from torch import nn
from torch.utils.data import DataLoader


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
        losses = self.load_losses()
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
                torch.save(self.model.state_dict(), self.model_file)
                print(f"  Saving updated model to {self.model_file}")
                if save_loss:  # only save updated losses if the model itself is being saved
                    self.save_losses(losses)
            else:
                stop_count += 1
                if stop_count >= early_stop:
                    print(f"  Exiting model training early after {epoch + 1} epochs")
                    return

    def save_losses(self, loss):
        with open(f"{self.model_file}.loss", "wb") as f:
            pickle.dump(loss, f)

    def load_losses(self):
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
            LayerClass = getattr(nn, layer["type"])
            self.layers.append(LayerClass(*layer.get("args", []), **layer.get("kwargs", {})))

    def forward(self, x):
        """
        Run a forward pass of the network
        Params:
          - x: The input variables
        """
        for layer in self.layers:
            x = layer(x)
        return x


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
        case _:
            raise ValueError(f"No model of type '{model_type}'")
