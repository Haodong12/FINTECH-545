import yfinance as yf
import pandas as pd

# Function to fetch data
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Function to prepare features
def prepare_features(stock_data, nasdaq_data):
    stock_data['Open'] = stock_data['Open']
    stock_data['High'] = stock_data['High']
    stock_data['Low'] = stock_data['Low']
    stock_data['MA5'] = stock_data['Close'].rolling(5).mean()
    stock_data['MA10'] = stock_data['Close'].rolling(10).mean()
    stock_data['MA20'] = stock_data['Close'].rolling(20).mean()  
    stock_data['MA50'] = stock_data['Close'].rolling(50).mean() 
    stock_data['pre_close'] = stock_data['Close'].shift(1)
    stock_data['price_change'] = stock_data['Close'] - stock_data['pre_close']
    stock_data['p_change'] = (stock_data['Close'] - stock_data['pre_close']) / stock_data['pre_close'] * 100
    stock_data['nasdaq'] = nasdaq_data['Close'].reindex(stock_data.index, method='nearest')
    stock_data['Target'] = stock_data['Close'].shift(-1)  
    stock_data.dropna(inplace=True)
    return stock_data

# Get data for a stock and S&P 500
stock_data = fetch_data('AAPL', '2012-01-01', '2023-12-31')
nasdaq_data = fetch_data('NQ=F', '2012-01-01', '2023-12-31')
prepared_data = prepare_features(stock_data, nasdaq_data)

# Save prepared data to a CSV file
prepared_data.to_csv('prepared_stock_data.csv')
