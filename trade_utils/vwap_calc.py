import yfinance as yf


def calculate_vwap(ticker):
    """
    Calculate the VWAP (Volume Weighted Average Price) for a given stock symbol.

    Parameters:
    stock_symbol (str): The ticker symbol for the stock (e.g., 'AAPL' for Apple Inc.).

    Returns:
    float: The calculated VWAP value.
    """
    # Download historical intraday data for the stock
    # Note: yfinance does not provide free intraday historical data.
    # For intraday data, you might need to use another API or service.
    df = yf.download(ticker, period="1d", interval="5m")

    # Check if the DataFrame is empty
    if df.empty:
        return "No data available for the specified stock."

    # Calculate the typical price for each period
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Calculate the price-volume
    df['PV'] = df['Typical_Price'] * df['Volume']

    # Calculate the cumulative total of price-volume and the cumulative total of volume
    df['Cumulative_PV'] = df['PV'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()

    # Calculate the VWAP
    vwap = df['Cumulative_PV'] / df['Cumulative_Volume']

    return vwap.iloc[-1]


# Example usage
ticker = "AAPL"
vwap_value = calculate_vwap(ticker)
print(f"VWAP for {ticker} is: {vwap_value}")
