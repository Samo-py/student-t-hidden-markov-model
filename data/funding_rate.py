import asyncio
import aiohttp
import pandas as pd

async def fetch_funding_rate(session, time):
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime={time}&limit=1000"
    async with session.get(url) as resp:
        return await resp.json()
        
async def get_funding_rate():
    time = 1546300800000 # Jan 1, 2019
    all_data = []
    async with aiohttp.ClientSession() as session:
        while True:
            data = await fetch_funding_rate(session, time)
            if not data:
                break

            all_data.extend(data)
            time = data[-1]["fundingTime"] + 1
            print("Funding rate fetched up to:", pd.to_datetime(time, unit="ms").date())

    dataframe = pd.DataFrame(all_data).rename(columns={"fundingTime": "date", "fundingRate": "funding_rate"}).drop(columns=["symbol", "markPrice"], axis=1)
    dataframe["date"] = pd.to_datetime(dataframe["date"], unit="ms").dt.normalize()
    dataframe = dataframe.set_index("date")
    dataframe["funding_rate"] = pd.to_numeric(dataframe["funding_rate"], errors="coerce")
    daily_df = dataframe.groupby("date")["funding_rate"].mean().to_frame()

    return daily_df

if __name__ == "__main__":
    df = asyncio.run(get_funding_rate())
    print(type(df.index[0]))