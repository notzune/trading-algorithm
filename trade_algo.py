import alpaca_trade_api as tradeapi
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Access the API key and secret from the environment
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Replace these with your Alpaca API key and secret
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Use the paper trading URL for testing

# Instantiate the Alpaca API connection
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)


def fetch_price_data(symbol, start_date, end_date, timeframe):
    """
    Fetch historical price data for a given stock symbol from Alpaca.
    """
    barset = api.get_bars(symbol, timeframe, start=start_date, end=end_date, limit=None, adjustment='raw').df
    return barset


def calculate_rsi(prices, period=14):
    """
    Calculate the RSI for the given price series.
    """
    rsi_indicator = RSIIndicator(prices['close'], window=period)
    return rsi_indicator.rsi()


def calculate_vwap(prices):
    """
    Calculate the VWAP for the given price series.
    """
    pv = prices['close'] * prices['volume']
    cumulative_pv = pv.cumsum()
    cumulative_volume = prices['volume'].cumsum()
    vwap = cumulative_pv / cumulative_volume
    return vwap


# Define your trading algorithm
def trade_logic(symbol):
    now = datetime.now()
    start_date = (now - timedelta(days=30)).isoformat(timespec='seconds') + 'Z'  # Add 'Z' to indicate UTC
    end_date = now.isoformat(timespec='seconds') + 'Z'  # Add 'Z' to indicate UTC

    # Fetch historical data
    prices = fetch_price_data(symbol, start_date, end_date, '1D')  # Use '1D' for daily bars

    # Calculate RSI and VWAP
    rsi = calculate_rsi(prices)
    vwap = calculate_vwap(prices)

    current_price = prices['close'].iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_vwap = vwap.iloc[-1]

    # Define the trading signals based on RSI and VWAP
    if last_rsi < 30 and current_price < last_vwap:
        print(f"Buying {symbol}, RSI is {last_rsi}, price below VWAP")
        # This is where you would place a buy order
        api.submit_order(symbol=symbol, qty=1, side='buy', type='market', time_in_force='gtc')
    elif last_rsi > 70 and current_price > last_vwap:
        print(f"Selling {symbol}, RSI is {last_rsi}, price above VWAP")
        # This is where you would place a sell order
        api.submit_order(symbol=symbol, qty=1, side='sell', type='market', time_in_force='gtc')


# Example usage
trade_logic('AAPL')
