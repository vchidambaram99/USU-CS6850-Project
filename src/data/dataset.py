import pandas as pd
import torch
from torch.utils.data import Dataset

from . import gen_dataset

STOCK_DAYS_PER_YEAR = 252  # Number of days stock market is typically open in one year
INPUT_DAYS = int(STOCK_DAYS_PER_YEAR / 2)
OUTPUT_DAYS = STOCK_DAYS_PER_YEAR - INPUT_DAYS


class BasicPreprocessor:
    """
    Preprocessor that min-max scales stock data between 0 and 1
    """

    def __init__(self):
        """
        Empty init function
        """
        pass

    def __call__(self, data, labels=None):
        """
        Preprocesses stock data by min-max scaling the columns of data and apply the transformation to labels
        Params:
          - data: Dataframe containing stock data
          - labels: Series containing labels (closing prices), or None for evaluation
        Return: (data as scaled numpy array, labels as scaled numpy array or None, parameters for unscaling)
        """
        maxes = data.max()
        mins = data.min()
        data = ((data - mins) / (maxes - mins)).to_numpy(dtype="float32")
        if labels is not None:
            labels = (labels - mins["close"]) / (maxes["close"] - mins["close"])
        return data, labels.to_numpy(dtype="float32"), {"min": mins["close"], "max": maxes["close"]}

    def unscaleLabels(self, labels, unscale_params: dict):
        """
        Unscales the labels (predicted or ground truth)
        Params:
          - labels: The labels to unscale
          - unscale_params: The parameters from unscaling (from the __call__ function)
        """
        if type(labels) is torch.Tensor:
            unscale_params = {k: v.reshape(*v.shape, 1) for k, v in unscale_params.items()}
        return (labels * (unscale_params["max"] - unscale_params["min"])) + unscale_params["min"]


class BasicStockData(Dataset):
    """
    Simple dataset that loads only the stock data collected in the data file and nothing supplementary
    """

    def __init__(self, data_file, preprocessor=BasicPreprocessor()):
        """
        Sets up dataset
        Params:
          - data_file: The parquet file to load data from
          - preprocessor: Preprocessor to transform data
        """
        self.data = pd.read_parquet(data_file)
        self.preprocessor = preprocessor
        self.pairs = []
        for stock in gen_dataset.get_tickers(self.data):
            self.pairs += [(stock, i) for i in range(0, len(self.data.loc[stock]) - STOCK_DAYS_PER_YEAR + 1)]

    def __len__(self):
        """
        Get the number of elements in the dataset
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a preprocessed element of the dataset by index
        """
        ticker = self.pairs[idx][0]
        start_idx = self.pairs[idx][1]
        stock_info = self.data.loc[ticker][start_idx : start_idx + STOCK_DAYS_PER_YEAR]
        label = stock_info[:OUTPUT_DAYS]["close"]
        data = stock_info[OUTPUT_DAYS:]
        return self.preprocessor(data, label)
