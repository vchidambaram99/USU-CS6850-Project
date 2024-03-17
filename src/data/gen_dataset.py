#!/usr/bin/env python3
import pandas as pd
import datetime
import requests
import json
import re
import time
import random
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Copied from https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)


def list_stocks(stock_file: str) -> list[str]:
    """
    Get the set of stock tickers provided in the file.
    Params:
      - stock_file: The file to get stock tickers from
    Returns: A set of stock tickers (duplicates removed)
    """
    stocks = set()
    with open(stock_file) as f:
        for line in f:
            stocks.add(line.strip())
    return list(stocks)


def stock_prices(ticker: str, start_date: datetime.date, end_date: datetime.date, asset_class="stocks") -> pd.DataFrame:
    """
    Grab a period of stock price data from the NASDAQ API.
    Params:
      - ticker: The stock ticker to get data for
      - start_date: The date to start getting data from
      - end_date: The date to get data up to
      - asset_class: The asset class to get data for (i.e. stocks, etfs, index)
    Returns: A table indexed by date of the information provided by the NASDAQ API
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"}
        limit = (end_date - start_date).days + 1
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        request = session.get(
            f"https://api.nasdaq.com/api/quote/{ticker}/historical?assetclass={asset_class}&fromdate={start_date}&todate={end_date}&limit={limit}",
            headers=headers,
        )
        data = json.loads(request.content)
        if data["data"] is None:
            print(f"Failed to get data for ticker {ticker}: {data}")
            return None
        table = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        clean_regex = re.compile("[$,]")
        as_float = lambda x: float(clean_regex.sub("", x))
        rows = data["data"]["tradesTable"]["rows"]
        table["close"] = pd.Series([as_float(r["close"]) for r in rows], index=[r["date"] for r in rows])
        table["open"] = pd.Series([as_float(r["open"]) for r in rows], index=[r["date"] for r in rows])
        table["high"] = pd.Series([as_float(r["high"]) for r in rows], index=[r["date"] for r in rows])
        table["low"] = pd.Series([as_float(r["low"]) for r in rows], index=[r["date"] for r in rows])
        table["volume"] = pd.Series([as_float(r["volume"]) for r in rows], index=[r["date"] for r in rows])
        return table
    except Exception as e:
        print(f"Failed to get data for ticker {ticker}: {e}")
        return None


def try_load_dataset(file: str):
    """
    Try to load a parquet dataset, returning an empty dataframe on failure
    Params:
      - file: The dataset to load
    Returns: Loaded dataframe, empty on failure
    """
    try:
        return pd.read_parquet(file)
    except Exception:
        return pd.DataFrame()


def get_tickers(df: pd.DataFrame):
    """
    Retrieve the tickers from a dataframe, if an index is named "ticker"
    Params:
      - df: The dataframe
    Returns: The tickers in the dataframe, empty list if none were found
    """
    idx = df.index
    if type(idx) is pd.Index and idx.name == "ticker":
        return list(idx)
    elif type(idx) is pd.MultiIndex and "ticker" in idx.names:
        return list(idx.levels[idx.names.index("ticker")])
    else:
        return []


def gen_dataset(
    stock_file: str, data_path: str, data_name: str, data_split={"train": 0.6, "valid": 0.2, "test": 0.2}
) -> None:
    """
    Generate a dataset using the provided stock file (should contain one stock ticker per line).
    Params:
      - stock_file: The file to get stock tickers from
      - data_path: Folder to store this dataset in
      - data_name: The name to use for this dataset
      - data_split: The split for the different subsets
    Notes: Will not alter or duplicate data already in datasets
    """
    # Get the list of stocks to download
    stocks = list_stocks(stock_file)

    # Load any old data so we don't download the same thing twice
    all_data = try_load_dataset(f"{data_path}/{data_name}_all.parquet")
    already_loaded = set([idx[0] for idx in all_data.index])  # All tickers already downloaded

    # For some reason, certain dates don't work with the API, just give it something more than 10 years ago
    start_date = datetime.date(2000, 1, 1)
    end_date = datetime.date.today()

    # Iterate through the stocks, downloading new stocks and adding them to the dataframe
    for i, ticker in enumerate(stocks):
        if ticker not in already_loaded:
            print(f"Retrieving data for stock {i}: {ticker}")
            table = stock_prices(ticker, start_date, end_date)
            if table is not None:
                all_data = pd.concat([all_data, pd.concat({ticker: table}, names=["ticker"])])
                if len(get_tickers(all_data)) % 20 == 0:  # Save every 20 stocks just in case
                    all_data.to_parquet(f"{data_path}/{data_name}_all.parquet")
            time.sleep(0.2)  # throttle downloading a little

    # Save all the downloaded data
    all_data.to_parquet(f"{data_path}/{data_name}_all.parquet")

    # Load any data already stored in datasets and filter out from newly downloaded
    stocks = set(get_tickers(all_data))
    datasets = {}
    for dataset_name in data_split.keys():
        datasets[dataset_name] = try_load_dataset(f"{data_path}/{data_name}_{dataset_name}.parquet")
        stocks -= set(get_tickers(datasets[dataset_name]))
    stocks = list(stocks)

    # Split the data into datasets
    random.shuffle(stocks)
    idx = 0
    for i, (dataset_name, dataset) in enumerate(datasets.items()):
        if i == len(datasets) - 1:  # get all remaining data for last dataset
            stock_split = stocks[idx:]
        else:
            new_idx = idx + int(data_split[dataset_name] * len(stocks))
            stock_split = stocks[idx:new_idx]
            idx = new_idx
        dataset = pd.concat([dataset, all_data.loc[stock_split]])
        dataset.to_parquet(f"{data_path}/{data_name}_{dataset_name}.parquet")

    # TODO create supplementary file with index data
    # df.to_parquet(f"{data_path}/{data_name}_supplementary.parquet")


def main():
    """
    Main function for gen_dataset.py, downloads data from NASDAQ based on command line arguments
    """
    parser = argparse.ArgumentParser(
        prog="gen_dataset", description="Downloads data from NASDAQ's historical stock price API"
    )
    parser.add_argument("stock_file", help="File containing stock tickers to download data for")
    parser.add_argument("data_path", help="Folder to place downloaded data into")
    parser.add_argument("dataset_name", help="The name of the dataset, used for creating output files")
    args = parser.parse_args()
    gen_dataset(args.stock_file, args.data_path, args.dataset_name)


if __name__ == "__main__":
    main()
