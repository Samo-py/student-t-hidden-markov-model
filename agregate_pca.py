import asyncio
import pandas as pd

import numpy as np


from data.dvol import dvol_deribit
from data.funding_rate import get_funding_rate
from data.price_volume import price_volume_binance


async def data_fetch(start, end):

    dvol_df = await dvol_deribit()
    funding_rate_df = await get_funding_rate()
    log_price_volume_df = await price_volume_binance()

    dataframe = pd.read_csv("data/btc_derivatives_history.csv")
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe = dataframe.set_index("date")


    df = pd.merge(pd.merge(pd.merge(dvol_df, funding_rate_df, how="inner", on="date"), log_price_volume_df, how="inner", on="date"), dataframe, on="date", how="inner")

    window = 30
    clip = 10
    


    '''window a clip je experimentalne, taktiez log funding a log IV'''

    # df["returns"] = np.log(df["price"] / df["price"].shift(1))
    # df["volume_z"] = (np.log(df["volume"]) - df["volume"].shift(1))
    # df["IV_z"] = df["dvol"]
    # df["funding_z"] = df["funding_rate"]
    # df["ls_z"] = df["ls_ratio"]
    # df["oi_z"] = np.log(df["oi_usd"] / df["oi_usd"].shift(1))

    df["returns_pct"] = np.log(df["price"] / df["price"].shift(1))
    df["oi_pct"] = np.log(df["oi_usd"] / df["oi_usd"].shift(1))


    df["returns_z"] = ((df["returns_pct"] - df["returns_pct"].mean()) / df["returns_pct"].rolling(window).std()).clip(-clip, clip)
    df["volume_z"] = ((np.log(df["volume"]) - df["volume"].rolling(window).mean()) / df["volume"].rolling(window).std()).clip(-clip, clip)
    df["IV_z"] = ((df["dvol"] - df["dvol"].rolling(window).mean()) / df["dvol"].rolling(window).std()).clip(-clip, clip)
    df["funding_z"] = ((df["funding_rate"] - df["funding_rate"].rolling(window).mean()) / df["funding_rate"].rolling(window).std()).clip(-clip, clip)
    df["ls_z"] = ((df["ls_ratio"] - df["ls_ratio"].rolling(window).mean()) / df["ls_ratio"].rolling(window).std()).clip(-clip, clip)
    df["oi_z"] = ((df["oi_pct"] - df["oi_pct"].rolling(window).mean()) / df["oi_pct"].rolling(window).std()).clip(-clip, clip)


    df = df[start:end]

    features = df[["returns_z", "volume_z", "IV_z", "funding_z", "ls_z", "oi_z"]].dropna()

    return features

if __name__ == "__main__":
    start = "2021-11-09"
    end = "2022-11-21"
    features = asyncio.run(data_fetch(start, end))
    print(features)