import asyncio
import aiohttp
import pandas as pd

import numpy as np

async def fetch_spot_price(session, time):
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={time}&limit=1000"
    async with session.get(url) as resp:
        return await resp.json()

async def get_spot_price():
    time = 1546300800000  # Jan 1, 2019
    all_data = []
    async with aiohttp.ClientSession() as session:
        while True:
            data = await fetch_spot_price(session, time)
            if not data:
                break
            all_data.extend(data)
            time = data[-1][6] + 1
            print("Price volume fetched up to:", pd.to_datetime(time, unit="ms").date())
            if len(data) < 1000:
                break

    dataframe = pd.DataFrame(all_data, columns=[
        "date", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    dataframe = dataframe.drop(columns=["close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms")
    dataframe = dataframe.set_index("date")
    for col in ["open", "high", "low", "close", "volume"]:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")

    return dataframe

async def price_volume_binance():
    df = await get_spot_price()
    df["price"] = df["close"]
    df = df.dropna().drop(columns=["open", "high", "low", "close"])
    return df

if __name__ == "__main__":
    print(asyncio.run(price_volume_binance()))