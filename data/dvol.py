import asyncio
import aiohttp
from datetime import datetime
import pandas as pd

url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

START_TS = int(datetime(2021, 3, 31).timestamp() * 1000)
END_TS = int(datetime.utcnow().timestamp() * 1000)

MAX_CANDLES = 1000

async def fetch_dvol(session, start, end):
    params = {
        "currency": "BTC",
        "start_timestamp": start,
        "end_timestamp": end,
        "resolution": "1D"
    }
    async with session.get(url, params=params) as resp:
        data = await resp.json()
        return data["result"]["data"]

async def get_dvol():
    all_rows = []
    cur_start = START_TS
    async with aiohttp.ClientSession() as session:
        while cur_start < END_TS:
            cur_end = min(cur_start + MAX_CANDLES * 24*60*60*1000, END_TS)
            candles = await fetch_dvol(session, cur_start, cur_end)
            if not candles:
                break

            for ts, o, h, l, c in candles:
                all_rows.append(
                    {
                        "date": ts,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c
                    }
                )
            
            cur_start = candles[-1][0] + 24*60*60*1000
            print("Dvol fetched up to:", datetime.utcfromtimestamp(candles[-1][0]/1000).date())

    dataframe = pd.DataFrame(all_rows)
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms").dt.normalize()
    dataframe = dataframe.set_index("date")
    return dataframe

async def dvol_deribit():
    df = await get_dvol()
    return df.drop(columns=["open", "high", "low"]).rename(columns={"close": "dvol"})

if __name__ == "__main__":
    df = asyncio.run(dvol_deribit())
    print(df)