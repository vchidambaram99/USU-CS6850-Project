#!/usr/bin/env python3
import argparse
import json
import pickle
from data import dataset
from models import models
from torch.utils.data import DataLoader


def train(args):
    """
    Subcommand function for training a model
    Params:
      - args: Namespace containing parsed arguments
    """
    model = models.load_model(args.model_config["model"], args.model_file)
    train_file = f"{args.data_path}/{args.dataset}_train.parquet"
    valid_file = f"{args.data_path}/{args.dataset}_valid.parquet"
    supp_file = f"{args.data_path}/{args.dataset}_supplementary.parquet"
    preprocs = args.model_config["data"]["preprocessors"]
    train_loader = DataLoader(dataset.StockData(train_file, supp_file, preprocs, sample=0.05), batch_size=64)
    valid_loader = DataLoader(dataset.StockData(valid_file, supp_file, preprocs, sample=0.05), batch_size=64)
    model.train(train_loader, valid_loader)


def test(args):
    model = models.load_model(args.model_config["model"], args.model_file)
    train_losses, valid_losses = model.load_losses()
    test_losses = {}
    train_file = f"{args.data_path}/{args.dataset}_train.parquet"
    valid_file = f"{args.data_path}/{args.dataset}_valid.parquet"
    test_file = f"{args.data_path}/{args.dataset}_test.parquet"
    supp_file = f"{args.data_path}/{args.dataset}_supplementary.parquet"
    preprocs = args.model_config["data"]["preprocessors"]
    train_loader = DataLoader(dataset.StockData(train_file, supp_file, preprocs), batch_size=64)
    valid_loader = DataLoader(dataset.StockData(valid_file, supp_file, preprocs), batch_size=64)
    test_loader = DataLoader(dataset.StockData(test_file, supp_file, preprocs), batch_size=64)
    test_losses["train"] = model.eval(train_loader)
    print(f"Model: {args.model_file}, train error: {test_losses['train']}")
    test_losses["valid"] = model.eval(valid_loader)
    print(f"Model: {args.model_file}, valid error: {test_losses['valid']}")
    test_losses["test"] = model.eval(test_loader)
    print(f"Model: {args.model_file}, test error: {test_losses['test']}")
    with open(f"{args.model_file}.loss", "wb") as f:
        pickle.dump({"train": train_losses, "valid": valid_losses, "test": test_losses}, f)


def read_model_config(file: str):
    """
    Read a json model configuration
    Params:
      - file: Path to the json model configuration
    Returns: Loaded json model configuration as dict
    """
    with open(file) as f:
        conf = json.load(f)
        conf["data"]["preprocessors"] = [dataset.getPreproc(**preproc) for preproc in conf["data"]["preprocessors"]]
        return conf


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
    parser.add_argument("--model-config", help="The configuration for the model", required=True, type=str)
    parser.add_argument("--model-file", help="Where to load model from or save model to")

    args = parser.parse_args()
    args.model_config = read_model_config(args.model_config)
    args.func(args)


if __name__ == "__main__":
    main()
