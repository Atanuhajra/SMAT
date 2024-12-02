import pandas as pd
import matplotlib.pyplot as plt
import nltk
import yfinance as yf
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import opinion_lexicon
from datetime import datetime, timedelta
import Sentiment


# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('opinion_lexicon')


# Example to ensure stock data is fetched correctly
stock_ticker = "^NSEI"  # NIFTY 50 Index. You can replace this with another stock symbol (e.g., 'AAPL')
start_date = sentiment_data['Date'].min() - timedelta(days=5)
end_date = sentiment_df['Date'].max() + timedelta(days=5)

# Fetch stock data using yfinance
try:
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"No stock data fetched for {stock_ticker} in the given range.")
except Exception as e:
    print(f"Error fetching stock data: {e}")
    stock_data = pd.DataFrame()  # Ensuring stock_data is an empty DataFrame on failure

# Make sure stock_data has the expected 'Date' column, and reset index if necessary
if not stock_data.empty:
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Ensure 'Date' in sentiment_df is in datetime format
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

    # Merge Sentiment Data with Stock Data based on Date
    merged_data = pd.merge(stock_data[['Date', 'Close']], sentiment_df, on='Date', how='inner')

    # Check if the merge was successful
    if not merged_data.empty:
        # Visualization - Dual Axis Plot
        plt.figure(figsize=(16, 8))

        # Plotting Stock Prices
        plt.plot(merged_data['Date'], merged_data['Close'], color='blue', label='Stock Close Price')

        # Plotting Sentiment Scores
        plt.twinx()  # Create a secondary y-axis
        plt.plot(merged_data['Date'], merged_data['Sentiment_Score'], color='green', label='Sentiment Score', alpha=0.6)

        # Add titles and labels
        plt.title('Stock Prices and Sentiment Analysis Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Stock Price', fontsize=14)
        plt.gca().set_ylabel('Sentiment Score', fontsize=14)

        # Add a legend
        plt.legend(loc='upper left')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("The merge resulted in no data. Please check the date range and ensure both data sources align.")
else:
    print("Stock data could not be fetched. Please check the ticker symbol and try again.")
