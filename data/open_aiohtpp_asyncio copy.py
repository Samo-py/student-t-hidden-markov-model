import asyncio
import aiohttp
import io
import zipfile
import pandas as pd

from datetime import datetime

SYMBOL = "BTCUSDT"
ARCHIVE = "https://data.binance.vision/data/futures/um/daily/metrics"

async def fetch_daily_csv(session: aiohttp.ClientSession, date : str) -> pd.DataFrame | None:
    try:
        url = f"{ARCHIVE}/{SYMBOL}/{SYMBOL}-metrics-{date}.zip"

        async with session.get(url) as resp:
            if resp.status == 404:
                print(f"{date} File does not exist")
                return None
            if resp.status != 200:
                print(f"{date} → HTTP {resp.status}")
                return None
            
            content = await resp.read()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
        
        return df

    except Exception as e:
        print(f"{date} → {type(e).__name__}: {e}")
        return None
        
async def fetch_oi_ls():
    records = []
    dates = pd.date_range("2021-09-01", pd.Timestamp.today() - pd.Timedelta(days = 1) , freq="D").astype("str")
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[fetch_daily_csv(session, date) for date in dates])

    for date, df in zip(dates, results):
        if df is None or df.empty:
            continue

        row = df.iloc[-1]
        records.append({
            "date" : date,
            "oi_usd" : row.get("sum_open_interest_value", None),
            "ls_ratio" : row.get("count_long_short_ratio", None)
        })

    df_out = pd.DataFrame(records)
    return df_out.sort_values("date").reset_index(drop=True)

if __name__ == "__main__":
    dataframe = asyncio.run(fetch_oi_ls())
    dataframe.to_csv("data/btc_derivatives_history.csv", index = False)

    print("Download complete")