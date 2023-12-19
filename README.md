# trading-algorithm

A python-based system designed to inform stock trading decisions by analyzing multiple financial indicators and market sentiments. 
It integrates a blend of technical analysis tools with sentiment analysis, leveraging a fine-tuned version of the FinBERT model to evaluate the potential of specific stocks.

--- 
## Components

The project is structured into different modules, each with a focused responsibility:

- `model_training`: Contains scripts for training machine learning models on market sentiment, utilizing FinBERT, a BERT-based model pre-trained on financial text.

- `trade_utils`: This directory houses utility scripts that perform various financial calculations:

  - `rsi_analysis.py`: Computes the Relative Strength Index (RSI) to identify overbought or oversold conditions.
  - `vwap_calc.py`: Calculates the Volume Weighted Average Price (VWAP) which helps to determine the market's true average price.
  - `short_potential_calc.py`: Evaluates the potential for a short squeeze based on short interest data.
  - `sentiment_analysis.py`: Applies sentiment analysis on news articles and social media posts concerning specific stocks.

## Project Goal

The primary objective is to develop an algorithm that can make informed trading decisions by considering both technical indicators and market sentiment. The algorithm aims to:

1. Aggregate and analyze news articles and social media posts for a given stock/ticker.
2. Utilize the FinBERT model to determine the sentiment conveyed in the media about the stock.
3. Calculate technical indicators like RSI and VWAP, and assess the short squeeze potential to understand market trends and pressures.
4. Compile these insights into a comprehensive dataset to establish a trading strategy, deciding when to buy or sell a stock.

## Future Enhancements

- Integration of sentiment scores with technical indicators to refine trading signals.
- Implementation of a backtesting framework to validate the strategy against historical data.
- Development of a real-time data pipeline for fetching the latest market and media information.
- Utilization of machine learning for predictive analysis and adaptive learning based on market conditions.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project was built as a hobby project by someone who has little to no knowledge about the stock market besides some light reading. The creator(s) and maintainer(s) of this project hold no liability in the event that you actually _use_ this algorithm for your own trading. This is strictly for educational and testing purposes **_ONLY_** and by no means should be used in an actual professional or commercial environment. 

---

This README is a living document and will evolve as the project grows and incorporates more features.
