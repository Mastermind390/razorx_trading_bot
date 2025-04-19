from typing import Final
from telegram import Bot
import asyncio
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("API_SECRET")
TELEGRAM = os.getenv("TELEGRAM")
CHAT_ID = os.getenv('CHAT_ID')

client = HTTP(demo=True, api_key=API_KEY,api_secret=SECRET_KEY)
TOKEN: Final = TELEGRAM
CHAT_ID: Final = CHAT_ID

bot = Bot(token=TOKEN)


def process_data(response):
    try:
        data = response["result"]["list"]
        column = ["time", "open", "high", "low","close", "volume", "turnover"]
        df = pd.DataFrame(data, columns = column)
        df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="ms")
        df.set_index('time', inplace=True)
        df = df.astype(float)
    except Exception as err:
        print(f"can't process data due to some error", {err})
    finally:
        return df   