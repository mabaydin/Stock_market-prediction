#pip install streamlit yfinance pandas numpy scikit-learn xgboost optuna plotly
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBClassifier
import optuna

st.set_page_config(page_title="Stock Prediction App", layout="wide")

st.title("Stock Prediction and Portfolio Tracker")

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to fetch live stock data
def fetch_live_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    return data

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

# Hyperparameter optimization function
def optimize_hyperparameters(X_train, y_train):
    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
        }
        
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        accuracy = accuracy_score(y_train, preds)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

# Sidebar for user inputs
st.sidebar.header("Stock Prediction")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL")
start_date = st.sidebar.date_input("Start date:", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date:", value=pd.to_datetime("2023-01-01"))

# Portfolio tracker
st.sidebar.header("Portfolio Tracker")
portfolio = st.sidebar.text_area("Enter your portfolio tickers (comma-separated):")
portfolio_list = [stock.strip().upper() for stock in portfolio.split(',')] if portfolio else []

if st.sidebar.button("Predict"):
    # Fetch and prepare data
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        st.error("No data fetched. Please check the ticker and date range.")
    else:
        features = create_features(data)
        X = features[['Open-Close', 'High-Low', 'SMA_5', 'SMA_10', 'SMA_15']]
        y = features['Increase']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Optimize hyperparameters
        best_params = optimize_hyperparameters(X_train, y_train)
        
        # Train an XGBoost model with the best parameters
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Accuracy: {accuracy * 100:.2f}%")
        st.write(f"Mean Squared Error: {mse}")
        
        # Predict whether to buy the stock today based on the latest data
        latest_data = create_features(data.tail(20)).iloc[-1:]
        latest_features = latest_data[['Open-Close', 'High-Low', 'SMA_5', 'SMA_10', 'SMA_15']]
        prediction = model.predict(latest_features)
        buy_signal = prediction[0] > 0.5
        
        if buy_signal:
            st.success(f"Based on the model, it is a good idea to buy {ticker} today.")
        else:
            st.warning(f"Based on the model, it is not a good idea to buy {ticker} today.")
        
        # Plot historical data
        fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Historical Prices')
        st.plotly_chart(fig)

# Display portfolio and live updates
if portfolio_list:
    st.header("Portfolio Tracker")
    portfolio_data = []
    for stock in portfolio_list:
        live_data = fetch_live_data(stock)
        if not live_data.empty:
            current_price = live_data['Close'].iloc[-1]
            portfolio_data.append({'Ticker': stock, 'Current Price': current_price})
        else:
            portfolio_data.append({'Ticker': stock, 'Current Price': 'N/A'})
    
    portfolio_df = pd.DataFrame(portfolio_data)
    st.table(portfolio_df)
