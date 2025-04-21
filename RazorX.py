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

tp = 0.05
sl = 0.01
mode = 1
leverage = "20"
qty = 20


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



def check_trend(symbol):
    data = get_kiline_data(client, symbol)
    close_prices = data.loc[:, 'close']
    ema_10 = calculate_ema(close_prices, 10)
    ema_20 = calculate_ema(close_prices, 20)
    fast_ema = ema_10[1]
    slow_ema = ema_20[1]
    latest_candle_open, latest_candle_high, latest_candle_low, latest_candle_close = get_candles_fo_check_trade_conditions(symbol)

    if fast_ema > slow_ema and all(value > fast_ema and value > slow_ema for value in [
    latest_candle_open, latest_candle_high, latest_candle_low, latest_candle_close]):
        return "uptrend"
    elif fast_ema < slow_ema and all(value < fast_ema and value < slow_ema for value in [
    latest_candle_open, latest_candle_high, latest_candle_low, latest_candle_close]):
        return "downtrend"


def get_tickers():
    high_volume_tickers = []

    try:
        tickers = client.get_tickers(category="linear")['result']['list']
        for ticker in tickers:
            symbol = ticker['symbol']
            turnover_24h = float(ticker['turnover24h'])  # This is in USD
            if turnover_24h >= 80000000.0000 and not "BTC" in symbol:
                high_volume_tickers.append(symbol)
        # print(high_volume_tickers)
        # print(len(high_volume_tickers))
        return high_volume_tickers
    except Exception as err:
        print(err)


def set_leverage(symbol):
    try:
        response = client.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=leverage,
            sellLeverage=leverage
        )
        return response
    except Exception as err:
        print(err)


def set_mode(symbol):
    try:
        resp = client.switch_margin_mode(
            category='linear',
            symbol=symbol,
            tradeMode = mode,
            buyLeverage=leverage,
            sellLeverage=leverage
        )
        # print(resp)
    except Exception as err:
        print(err)


def get_price_precision(symbol):
    response = client.get_instruments_info(category="linear", symbol=symbol)["result"]["list"][0]
    price = response["priceFilter"]["tickSize"]
    
    if "." in price:
        price = len(price.split(".")[1])
    else:
        price = 0

    quantity = response["lotSizeFilter"]["qtyStep"]

    if "." in quantity:
        quantity = len(quantity.split(".")[1])
    else:
        quantity = 0
    
    return price, quantity

def take_profit_and_stop_loss(symbol):
    pen, high, low, close = get_candles_fo_check_trade_conditions(symbol)

    price_precision = get_price_precision(symbol)[0]
    qty_precision = get_price_precision(symbol)[1]
    buy_tpPrice = round(close + close * tp, price_precision)
    buy_slPrice = round(close - close * sl, price_precision)
    sell_tpPrice = round(close - close * tp, price_precision)
    sell_slPrice = round(close + close * sl, price_precision)
    order_qty = round(qty/close, qty_precision)
    entry_price =  round(close, price_precision)

    return buy_tpPrice, buy_slPrice, sell_tpPrice, sell_slPrice, order_qty, entry_price


def place_buy_order(symbol):
    try:
        open, high, low, close = get_candles_fo_check_trade_conditions(symbol)

        buy_tpPrice, buy_slPrice, sell_tpPrice, sell_slPrice, order_qty, entry_price = take_profit_and_stop_loss(symbol)

        response = client.place_order(
                category="linear",
                symbol=symbol,
                side="Buy",
                orderType="Limit",
                qty=order_qty,
                price = entry_price,
                takeProfit = buy_tpPrice,
                time_in_force="GTC"
            )
        print(f"BUY order placed for: {symbol}")
    except Exception as err:
        print(err)


def place_sell_order(symbol):
    try:
        open, high, low, close = get_candles_fo_check_trade_conditions(symbol)
        
        buy_tpPrice, buy_slPrice, sell_tpPrice, sell_slPrice, order_qty, entry_price = take_profit_and_stop_loss(symbol)

        response = client.place_order(
                category="linear",
                symbol=symbol,
                side="Sell",
                orderType="Limit",
                qty=order_qty,
                price = entry_price,
                takeProfit = sell_tpPrice,
                time_in_force="GTC"
            )
        print(f"SELL order placed for: {symbol}")
    except Exception as err:
        print(err)

async def send_signal(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)


sendSignal = False

sent_signals = []

symbol_signal = {}

async def check_trade_conditions(symbol):
    global sendSignal

    trend = check_trend(symbol)
    rsi = calculate_rsi(symbol)

    open, high, low, close = get_candles_fo_check_trade_conditions(symbol)

    buy_tpPrice, buy_slPrice, sell_tpPrice, sell_slPrice, order_qty, entry_price = take_profit_and_stop_loss(symbol)

    # Find existing signal if any
    existing_signal = next((s for s in sent_signals if s["symbol"] == symbol), None)

    # Only proceed if no signal exists, or signal exists but not yet sent
    if not existing_signal or not existing_signal["signal_sent"]:

        if trend == "uptrend" and rsi >= 30:
            place_buy_order(symbol)
            BUY_SIGNAL = (
                f"🚨 Trade Alert for {symbol} 🚨\n\n"
                f"Signal: BUY 🟢\n"
                f"Entry Price: {entry_price}\n"
                f"Take Profit: {buy_tpPrice}\n"
                f"Stop Loss: {buy_slPrice}\n"
                f"🔔 Stay sharp and manage your risk!"
            )
            new_signal = {
                "signal_sent": True,
                "symbol": symbol,
                "entry_price": entry_price,
                "buy_tpPrice": buy_tpPrice,
                "buy_slPrice": buy_slPrice,
                "order_type": "buy"
            }
            if existing_signal:
                sent_signals.remove(existing_signal)
            sent_signals.append(new_signal)
            print(BUY_SIGNAL)

        elif trend == "downtrend" and rsi <= 70:
            place_sell_order(symbol)
            SELL_SIGNAL = (
                f"🚨 Trade Alert for {symbol} 🚨\n\n"
                f"Signal: SELL 🔻\n"
                f"Entry Price: {entry_price}\n"
                f"Take Profit: {sell_tpPrice}\n"
                f"Stop Loss: {sell_slPrice}\n"
                f"🔔 Stay sharp and manage your risk!"
            )
            new_signal = {
                "signal_sent": True,
                "symbol": symbol,
                "entry_price": entry_price,
                "buy_tpPrice": sell_tpPrice,
                "buy_slPrice": sell_slPrice,
                "order_type": "sell"
            }
            if existing_signal:
                sent_signals.remove(existing_signal)
            sent_signals.append(new_signal)
            print(SELL_SIGNAL)

        else:
            print("no signal")
    else:
        print("signal already sent and marked as sent")

    print(sent_signals)


    
    # if sendSignal and close > entry_price and close >= buy_tpPrice:
    #     BUY_TP_SIGNAL = (
    #         f"🎯 TP Hit Alert for {symbol} 🎯\n\n"
    #         f"Congratulations!!! 🎆🎇\n"
    #         f"TP Hit at: {buy_tpPrice}\n"
    #         f"More more wins guys 🎈📊"
    #     )
    #     sendSignal = False
    #     print(BUY_TP_SIGNAL)
    # elif sendSignal and close < entry_price and close  <= buy_slPrice:
            # BUY_SL_SIGNAL = (
            #     f"😢 SL Hit Alert for {symbol} 😢\n\n"
            #     f"losses is part of the game. keep moving guys\n"
            #     f"SL Hit at: {buy_slPrice}\n"
            #     f"💔💔💔💔"
            # )
            # sendSignal = False
            # print(BUY_SL_SIGNAL)



# print(symbols)


async def main():
    symbols = get_tickers()
    while True:
        for symbol in symbols:
            set_leverage(symbol)
            set_mode(symbol)
            try:
                await check_trade_conditions(symbol)
            except Exception as err:
                print(f"Error checking signal for {symbol}: {err}")
        await asyncio.sleep(900)

if __name__ == "__main__":
    asyncio.run(main())