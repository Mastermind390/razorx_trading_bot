from typing import Final
from telegram import Bot
import asyncio
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import pandas as pd
import numpy as np
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


def get_kiline_data(client, symbol, interval = 15):
    try:
        response = client.get_kline(category="linear",symbol=symbol,interval="15")
    except:
        print("error getting data")
    finally:
        return process_data(response)


def calculate_ema(prices, period):
    ema_values = []
    multiplier = 2 / (period + 1)

    # Start with SMA
    sma = sum(prices[:period]) / period
    ema_values.append(sma)

    for price in prices[period:]:
        ema_prev = ema_values[0]
        ema_current = (price - ema_prev) * multiplier + ema_prev
        ema_values.append(ema_current)

    return ema_values

data = get_kiline_data(client, "BTCUSDT")
close_prices = data.loc[:, 'close']
ema_10 = calculate_ema(close_prices, 10)
ema_20 = calculate_ema(close_prices, 20)
print(ema_20)


def calculate_rsi(symbol, period=14, column='close'):

    data = get_kiline_data(client, symbol, interval = 15)

    delta = data.loc[:, column].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average (EMA)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    semi_rsi = 100 - (100 / (1 + rs)).dropna()
    final_rsi = semi_rsi.to_numpy()
    rsi = final_rsi[1] + 3.6

    return rsi


def get_candles_fo_check_trade_conditions(symbol):
    candles = get_kiline_data(client, symbol, interval = 15)

    latest_candle_open = candles.iloc[1, 0]
    latest_candle_high = candles.iloc[1, 1]
    latest_candle_low = candles.iloc[1, 2]
    latest_candle_close = candles.iloc[1, 3]

    return latest_candle_open, latest_candle_high, latest_candle_low, latest_candle_close




# if fast_ema > slow_ema and latest_candle_open > fast_ema and latest_candle_high > fast_ema and latest_candle_low > fast_ema and latest_candle_close > fast_ema and latest_candle_open > slow_ema and latest_candle_high > slow_ema and latest_candle_low > slow_ema and latest_candle_close > slow_ema:

def check_trend(symbol):
    ema_10, ema_20 = calculate_ema(symbol)
    fast_ema = ema_10
    slow_ema = ema_20
    latest_candle_open, latest_candle_high, latest_candle_low, latest_candle_close = get_candles_fo_check_trade_conditions(symbol)

    # if fast_ema > slow_ema and latest_candle_close > fast_ema and latest_candle_close > slow_ema:
    #     print("up")
    # else:
    #     print("down")

    print(fast_ema, slow_ema)
    

# check_trend("ETHUSDT")

# rsi = calculate_rsi(symbol)

# and all(value > fast_ema and value > slow_ema for value in [
#     latest_candle_open, latest_candle_high, latest_candle_low, latest_candle_close]):
#         print("uptend")
#     elif fast_ema < slow_ema and all(value < fast_ema and value < slow_ema for value in [
#     latest_candle_open, latest_candle_high, latest_candle_low, latest_candle_close]):
#         print("downtrend")