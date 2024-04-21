import polars as pl
import torch
import random
from torch.utils.data import Dataset


INPUT_DAYS = 126  # Number of days stock market is open in 6 months
OUTPUT_DAYS = 21  # Average number of days stock market is open in 1 month
TOTAL_DAYS = INPUT_DAYS + OUTPUT_DAYS


def clean(e: pl.Expr, default: float = 0):
    """
    Clean an expression, removing NaNs and infinities
    Params:
      - e: An expression to clean
      - default: The value to set in place of bad values
    """
    e = e.fill_nan(default)
    return pl.when(e.is_infinite()).then(default).otherwise(e)


def minMaxScaler(column: str):
    """
    Returns a polars expression for preprocessing a column with min-max scaling
    Params:
      - column: column name
    Return: Polars expression for min-max scaling the column
    """
    col = pl.col(column)
    return clean((col - col.min()) / (col.max() - col.min()), default=0.5).alias(column)


def proportionalDiff(column: str):
    """
    Returns a polars expression for preprocessing a column with proportional difference from end
    Params:
      - column: column name
    Return: Polars expression for proportional diff scaling the column
    """
    col = pl.col(column)
    return clean((col / col.last()) - 1).alias(column)


def proportionalDiffConsecutive(column: str):
    """
    Returns a polars expression for preprocessing a column with proportional difference from previous
    Params:
      - column: column name
    Return: Polars expression for consecutive proportional diff scaling the column
    """
    col = pl.col(column)
    return clean((col / col.shift(fill_value=col.first())) - 1).alias(column)


def getPreproc(name: str, args: list):
    """
    Get a preprocessing expression
    Params:
      - name: The name of the preprocessor function
      - args: The list of inputs to the preprocessor function
    Return: A preprocessing expression (or raises error if none matched)
    """
    functions = {"MinMaxScaler": minMaxScaler, "ProportionalDiff": proportionalDiff, "ProportionalDiffConsecutive": proportionalDiffConsecutive}
    if name in functions:
        return functions[name](*args)
    else:
        raise ValueError(f"No preprocessor found with name {name}")


def doubleToFloat(df: pl.DataFrame):
    for col in df.columns:  # Cast all float64 data to float32
        if df[col].dtype == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
    return df


class StockData(Dataset):
    """
    Dataset that loads the stock data and supplementary data according to the provided options
    """

    def __init__(self, data_file: str, supp_file: str, preprocessors: list[tuple], sample=1.0):
        """
        Sets up dataset
        Params:
          - data_file: The parquet file to load data from
          - supp_file: The parquet file containing supplementary data
          - preprocessors: List of columns and functions for preprocessing them
          - sample: The proportion of data to use each epoch
        """
        self.data = doubleToFloat(pl.read_parquet(data_file).reverse())  # put dates in ascending order
        self.preprocessors = preprocessors
        self.sample = sample
        counts = self.data.group_by("ticker").agg(pl.count())
        self.offsets = []
        for ticker, count in counts.rows():
            offset = self.data.with_row_index().filter(pl.col("ticker") == ticker)["index"][0]
            self.offsets += [offset + i for i in range(0, count - TOTAL_DAYS + 1)]

        supp_table = doubleToFloat(pl.read_parquet(supp_file).reverse())  # put dates in ascending order
        self.supp_data = {}
        self.supp_data_indices = {}
        for ticker in supp_table["ticker"].unique():
            df = supp_table.filter(pl.col("ticker") == ticker)
            indices = {}
            for idx, date in enumerate(df["date"]):
                indices[date] = idx
            self.supp_data_indices[ticker] = indices
            self.supp_data[ticker] = df.rename(lambda n: f"{ticker}_{n}").lazy()

        self.shuffle()

    def __len__(self):
        """
        Get the number of elements in the dataset
        """
        return int(len(self.offsets) * self.sample)

    def __getitem__(self, idx):
        """
        Get a preprocessed element of the dataset by index
        """
        offset = self.offsets[idx]
        stock_info = self.data[offset : offset + INPUT_DAYS]
        stock_info_lazy = stock_info.lazy()
        end_close = stock_info["close"][-1]
        labels = (self.data["close"][offset + INPUT_DAYS : offset + TOTAL_DAYS].to_numpy().flatten() / end_close) - 1

        for ticker, frame in self.supp_data.items():
            supp_offset = self.supp_data_indices[ticker][stock_info["date"][0]]
            stock_info_lazy = pl.concat([stock_info_lazy, frame[supp_offset : supp_offset + INPUT_DAYS]], how="horizontal")

        out_df = stock_info_lazy.select(*self.preprocessors).collect()
        for column in out_df.columns:
            if out_df[column].is_nan().any():
                print(f"Null in column {column}")
                print(stock_info_lazy.collect())
        return out_df.to_numpy().transpose(), labels, {"scale": end_close}

    def shuffle(self):
        """
        Shuffle the data so the sample refers to different data
        """
        random.shuffle(self.offsets)

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
