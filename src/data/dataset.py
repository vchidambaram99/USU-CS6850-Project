import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from . import gen_dataset

INPUT_DAYS = 126  # Number of days stock market is open in 6 months
OUTPUT_DAYS = 21  # Average number of days stock market is open in 1 month
TOTAL_DAYS = INPUT_DAYS + OUTPUT_DAYS


def minMaxScaler(data: np.array):
    """
    Preprocesses data by min-max scaling
    Params:
      - data: Array containing stock data
    Return: Array of scaled data
    """
    max = np.max(data)
    min = np.min(data)
    return (data - min) / (max - min)


def proportionalDiff(data: np.array):
    """
    Preprocesses data by proportional difference from start
    Params:
      - data: Array containing stock data
    Return: Array of scaled data
    """
    return (data / data[0]) - 1


def proportionalDiffConsecutive(data: np.array):
    """
    Preprocesses data by proportional difference from the previous value
    Params:
      - data: Array containing stock data
    Return: Array of scaled data
    """
    return np.concatenate(([0], (data[1:] / data[:-1]) - 1))


def getPreproc(name: str):
    """
    Get one of the preprocessor functions by name
    Params:
      - name: The name of the preprocessor function
    Return: The specified function (or raises error if none matched)
    """
    if name == "MinMaxScaler":
        return minMaxScaler
    elif name == "ProportionalDiff":
        return proportionalDiff
    elif name == "ProportionalDiffConsecutive":
        return proportionalDiffConsecutive
    else:
        raise ValueError(f"No preprocessor found with name {name}")


class StockData(Dataset):
    """
    Dataset that loads the stock data and supplementary data according to the provided options
    """

    def __init__(self, data_file: str, supp_file: str, preprocessors: list[tuple]):
        """
        Sets up dataset
        Params:
          - data_file: The parquet file to load data from
          - supp_file: The parquet file containing supplementary data
          - preprocessors: List of columns and functions for preprocessing them
        """
        self.data = pd.read_parquet(data_file)
        self.preprocessors = preprocessors
        self.pairs = []
        for stock in gen_dataset.get_tickers(self.data):
            self.pairs += [(stock, i) for i in range(0, len(self.data.loc[stock]) - TOTAL_DAYS + 1)]
        supp_table = pd.read_parquet(supp_file)
        self.supp_data = {}
        for ticker in gen_dataset.get_tickers(supp_table):
            self.supp_data[ticker] = supp_table.loc[ticker]

    def __len__(self):
        """
        Get the number of elements in the dataset
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a preprocessed element of the dataset by index
        """
        ticker, start_idx = self.pairs[idx]
        stock_info = self.data.loc[ticker][start_idx : start_idx + TOTAL_DAYS][::-1]  # reverse order so low index is older
        labelScale = stock_info.iloc[0]["close"]
        label = (stock_info[INPUT_DAYS:]["close"] / labelScale) - 1
        data = stock_info[:INPUT_DAYS]
        supp_data = {}
        for ticker, table in self.supp_data.items():
            loc = table.index.get_loc(data.index[0])
            supp_data[ticker] = table[loc : loc + INPUT_DAYS][::-1]
        return self.processData(data, supp_data), label.to_numpy(dtype=np.float32), {"scale": labelScale}

    def processData(self, data: pd.DataFrame, supp: dict[str, pd.DataFrame]):
        """
        Apply preprocessing functions to given data
        Params:
          - data: Dataframe containing the data to preprocess
          - supp: dictionary of tickers to supplemental DataFrames for preprocessing
        Returns: Preprocessed data (input for model)
        """
        out = np.zeros((len(self.preprocessors), INPUT_DAYS), dtype=np.float32)
        for i, (feature_name, preprocFunc) in enumerate(self.preprocessors):
            if "_" in feature_name:  # supplemental
                ticker, feature = feature_name.split("_")
                out[i, :] = preprocFunc(supp[ticker][feature])
            else:
                out[i, :] = preprocFunc(data[feature_name])
        if out.shape[0] == 1:
            out = out.reshape(out.shape[1])
        return out

    def unscaleLabels(self, labels, unscale_params: dict):
        """
        Unscales the labels (predicted or ground truth)
        Params:
          - labels: The labels to unscale
          - unscale_params: The parameters from unscaling (from the __call__ function)
        """
        if type(labels) is torch.Tensor:
            unscale_params = {k: v.reshape(*v.shape, 1) for k, v in unscale_params.items()}
        return (labels + 1) * unscale_params["scale"]
