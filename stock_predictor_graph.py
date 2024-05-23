import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import optuna

st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("Stock Prediction App")

# Top 50 stocks list
top_50_stocks = [
    ('AAPL', 'Apple Inc.'), ('MSFT', 'Microsoft Corporation'), ('GOOGL', 'Alphabet Inc.'), ('AMZN', 'Amazon.com Inc.'), 
    ('TSLA', 'Tesla Inc.'), ('META', 'Meta Platforms, Inc.'), ('NVDA', 'NVIDIA Corporation'), ('BRK-B', 'Berkshire Hathaway Inc.'), 
    ('JPM', 'JPMorgan Chase & Co.'), ('JNJ', 'Johnson & Johnson'), ('V', 'Visa Inc.'), ('PG', 'Procter & Gamble Company'), 
    ('UNH', 'UnitedHealth Group Incorporated'), ('HD', 'The Home Depot, Inc.'), ('MA', 'Mastercard Incorporated'), 
    ('DIS', 'The Walt Disney Company'), ('PYPL', 'PayPal Holdings, Inc.'), ('BAC', 'Bank of America Corporation'), 
    ('VZ', 'Verizon Communications Inc.'), ('ADBE', 'Adobe Inc.'), ('CMCSA', 'Comcast Corporation'), ('NFLX', 'Netflix, Inc.'), 
    ('KO', 'The Coca-Cola Company'), ('PFE', 'Pfizer Inc.'), ('T', 'AT&T Inc.'), ('PEP', 'PepsiCo, Inc.'), 
    ('XOM', 'Exxon Mobil Corporation'), ('CSCO', 'Cisco Systems, Inc.'), ('MRK', 'Merck & Co., Inc.'), ('ABT', 'Abbott Laboratories'), 
    ('INTC', 'Intel Corporation'), ('CVX', 'Chevron Corporation'), ('NKE', 'NIKE, Inc.'), ('WMT', 'Walmart Inc.'), 
    ('LLY', 'Eli Lilly and Company'), ('TMO', 'Thermo Fisher Scientific Inc.'), ('ORCL', 'Oracle Corporation'), 
    ('MDT', 'Medtronic plc'), ('ACN', 'Accenture plc'), ('HON', 'Honeywell International Inc.'), ('AVGO', 'Broadcom Inc.'), 
    ('MCD', "McDonald's Corporation"), ('COST', 'Costco Wholesale Corporation'), ('DHR', 'Danaher Corporation'), 
    ('TXN', 'Texas Instruments Incorporated'), ('NEE', 'NextEra Energy, Inc.'), ('QCOM', 'QUALCOMM Incorporated'), 
    ('UPS', 'United Parcel Service, Inc.'), ('PM', 'Philip Morris International Inc.'), ('BMY', 'Bristol-Myers Squibb Company')
]

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Feature engineering function
def create_features(df):
    df = df.copy()
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df['Increase'] = (df['Close'].shift(-1) - df['Close']).apply(lambda x: 1 if x > 0 else 0)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_15'] = df['Close'].rolling(window=15).mean()
    df.dropna(inplace=True)
    return df[['Open-Close', 'High-Low', 'SMA_5', 'SMA_10', 'SMA_15', 'Increase']]

# Sidebar for user inputs
st.sidebar.header("Stock Prediction")
start_date = st.sidebar.date_input("Start date:", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date:", value=pd.to_datetime("today"))

if st.sidebar.button("Check All Stocks"):
    results = []

    for ticker, name in top_50_stocks:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            continue

        features = create_features(data)
        X = features[['Open-Close', 'High-Low', 'SMA_5', 'SMA_10', 'SMA_15']]
        y = features['Increase']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        latest_data = create_features(data.tail(20)).iloc[-1:]
        latest_features = latest_data[['Open-Close', 'High-Low', 'SMA_5', 'SMA_10', 'SMA_15']]
        prediction = model.predict(latest_features)
        buy_signal = prediction[0] > 0.5
        
        status = "Yes" if buy_signal else "No"
        results.append({"Date": end_date, "Ticker": ticker, "Name": name, "Buy Today": status, "Accuracy": accuracy})
    
    results_df = pd.DataFrame(results)
    
    # Change color of Buy Today column
    def highlight(status):
        color = 'green' if status == 'Yes' else 'red'
        return f'color: {color}'
    
    results_df_styled = results_df.style.applymap(highlight, subset=['Buy Today'])
    
    st.write(results_df_styled)
