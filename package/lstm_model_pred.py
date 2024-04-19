import pandas as pd
import numpy as np
import os
import sys
import joblib

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime, timedelta


def doItAll(a, b, c):
    print("abc")
    print(a, b, c)
    trainStock = a
    # For testing purposes
    trainStock = "AAPL"

    trainStartDate = b
    # For testing purposes
    trainStartDate = "2020-01-01"

    df2 = pdr.get_data_yahoo(trainStock, start=trainStartDate, end=datetime.now())
    # Show the data
    # Create a new dataframe with only the 'Close column
    data = df2.filter(["Close"])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * 0.95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0 : int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print("61")

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)
    test_data = scaled_data[training_data_len - 60 :, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    argList = []
    for arg in sys.argv[1:]:
        argList.append(arg)

    predictStock = c

    # Now to do a specific prediction
    stock_quote = pdr.get_data_yahoo(
        predictStock, start="2024-01-01", end=datetime.now()
    )
    new_df = stock_quote.filter(["Close"])
    last_60_days = new_df[-60:].values
    # Scale the data to be values between 0
    last_60_days_scaled = scaler.transform(last_60_days)

    # Create an empty list
    pred_list = []
    # Appemd the past 60days
    pred_list.append(last_60_days_scaled)

    # Conver the pred_list data into numpy array
    pred_list = np.array(pred_list)

    # Reshape the data
    pred_list = np.reshape(pred_list, (pred_list.shape[0], pred_list.shape[1], 1))
    # Get predicted scaled price
    pred_price = model.predict(pred_list)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    print(f"Price of {predictStock} tomorrow:{pred_price}")
    return pred_price
