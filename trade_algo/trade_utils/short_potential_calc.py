import yfinance as yf


def get_short_squeeze_potential(ticker):
    # Fetch data for the given ticker symbol
    stock = yf.Ticker(ticker)

    # Get necessary data points
    data = stock.info
    short_ratio = data.get('shortRatio')
    float_shares = data.get('floatShares')
    shares_short = data.get('sharesShort')
    short_percent_float = data.get('shortPercentOfFloat')
    average_daily_volume = data.get('averageDailyVolume10Day')

    # Calculate Days to Cover (Short Interest Ratio)
    days_to_cover = None
    if short_ratio and average_daily_volume:
        days_to_cover = shares_short / average_daily_volume

    # Determine the potential for a short squeeze
    potential = "Unknown"
    if short_percent_float and short_percent_float > 0.1:  # Example threshold
        if days_to_cover and days_to_cover > 5:  # Example threshold
            potential = "High"
        else:
            potential = "Medium"
    else:
        potential = "Low"

    return {
        'Ticker': ticker,
        'Short Ratio': short_ratio,
        'Days to Cover': days_to_cover,
        'Short Percent of Float': short_percent_float,
        'Potential for Short Squeeze': potential
    }


# Example usage
ticker = 'GME'  # Example ticker with historical short squeeze
potential = get_short_squeeze_potential(ticker)
print(potential)
