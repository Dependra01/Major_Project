#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error
import json
import locale
import os

# Set stdout to UTF-8 for proper encoding on Windows
sys.stdout.reconfigure(encoding='utf-8')

# combined_args = sys.argv[1]
combined_args = sys.argv[1] if len(sys.argv) > 1 else ""
if not combined_args:
    print(json.dumps({"error": "No arguments provided"}))
    sys.exit(1)

# Parse arguments
try:
    start, end, stock_symbol = combined_args.split(',')
    print(f"Start: {start}, End: {end}, Stock Symbol: {stock_symbol}")
except ValueError:
    print(json.dumps({"error": "Invalid input format. Expected: start,end,stock_symbol"}))
    sys.exit(1)

def predict_stock_prices(start, end, stock_symbol, ttldays=30):
    # Download stock data
    try:
        df = yf.download(stock_symbol + ".NS", start=start, end=end)
        if df.empty:
            print(json.dumps({"error": f"No data found for {stock_symbol}"}))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Error downloading data: {str(e)}"}))
        sys.exit(1)

    # Prepare data
    df1 = df.reset_index()['Close']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Split data into training and test sets
    training_size = int(len(df1) * 0.75)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size], df1[training_size:len(df1)]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            dataX.append(dataset[i:(i + time_step), 0])
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = min(100, test_size - 5)
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input data for LSTM model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define and compile LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Predict future stock prices
    x_input = test_data[-time_step:].reshape(1, time_step, 1)
    predictions = []

    for i in range(ttldays):
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0, 0])
        x_input = np.append(x_input, yhat.reshape(1, 1, 1), axis=1)
        x_input = x_input[:, 1:, :]

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions.tolist()

# Call the function and print predictions
try:
    predictions = predict_stock_prices(start, end, stock_symbol)
    print(json.dumps(predictions))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
