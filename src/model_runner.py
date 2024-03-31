#!/usr/bin/env python3
import argparse
import json
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
    pass


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
