import yfinance as yf
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta


def get_stock_rsi(ticker, period=14, days_back=365):
    """
    Calculate the RSI (Relative Strength Index) for a given stock symbol and determine its trading condition.

    Parameters:
    stock_symbol (str): The ticker symbol for the stock (e.g., 'AAPL' for Apple Inc.).
    period (int): The period to calculate RSI, typically 14 days.
    days_back (int): The number of days in the past to start fetching data from.

    Returns:
    str: A message indicating the RSI value and whether the stock is overbought, oversold, or neutral.
    """

    # Determine the current date and the start date
    end_date = datetime.now().strftime('%Y-%m-%d')  # Format current date as YYYY-MM-DD
    start_date = (datetime.now() - timedelta(days=days_back)).strftime(
        '%Y-%m-%d')  # Start date set to 'days_back' days ago

    # Download historical stock data using yfinance
    df = yf.download(ticker, start=start_date, end=end_date)

    # Check if the DataFrame is empty
    if df.empty:
        return "No data available for the specified stock."

    # Calculate RSI
    rsi_indicator = RSIIndicator(df['Close'], window=period)
    df['RSI'] = rsi_indicator.rsi()

    # Get the last RSI value
    last_rsi = df['RSI'].iloc[-1]

    # Determine overbought/oversold
    if last_rsi > 70:
        return f"RSI is {last_rsi:.2f}, indicating that the stock may be overbought."
    elif last_rsi < 30:
        return f"RSI is {last_rsi:.2f}, indicating that the stock may be oversold."
    else:
        return f"RSI is {last_rsi:.2f}, indicating neutral conditions."


# Example usage
ticker = "AAPL"
print(get_stock_rsi(ticker))
