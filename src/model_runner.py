#!/usr/bin/env python3
import argparse
from data import dataset
from models import models
from torch.utils.data import DataLoader


def train(args):
    """
    Subcommand function for training a model
    Params:
      - args: Namespace containing parsed arguments
    """
    model = models.load_model(args.model_type, args.model_file)
    train_loader = DataLoader(
        dataset.BasicStockData(f"{args.data_path}/{args.dataset}_train.parquet"), batch_size=64, shuffle=True
    )
    valid_loader = DataLoader(dataset.BasicStockData(f"{args.data_path}/{args.dataset}_valid.parquet"), batch_size=64)
    model.train(train_loader, valid_loader)


def test(args):
    pass


def main():
    """
    Main function for the model runner, parses arguments, calls function for subcommand
    """
    parser = argparse.ArgumentParser(prog="model_runner", description="Train/evaluate a neural network model")
    subparsers = parser.add_subparsers(required=True)
    parser_train = subparsers.add_parser("train", description="Train a new or existing model")
    parser_train.set_defaults(func=train)
    parser_test = subparsers.add_parser("test", description="Test an existing model")
    parser_test.set_defaults(func=test)
    parser.add_argument("--data-path", help="Where the data is located", required=True)
    parser.add_argument("--dataset", help="The name of the dataset", required=True)
    parser.add_argument("--model-type", help="The type of the model")
    parser.add_argument("--model-file", help="Where to load model from or save model to")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
