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

# argList = []
# for arg in sys.argv[1:]:
#    argList.append(arg)

# predictStock = argList[0]


def getPrediction(predictStock):
    a = os.listdir(path="./models")
    predictStock = predictStock.upper()
    model = keras.models.load_model(rf"models/model.keras")
    scaler = joblib.load(rf"models/scaler.gz")
    if f"{predictStock}.keras" in a:
        scaler = joblib.load(rf"models/{predictStock}.gz")
        model = keras.models.load_model(rf"models/{predictStock}.keras")

    # Now to do a specific prediction
    stock_quote = pdr.get_data_yahoo(
        predictStock, start="2023-01-01", end=datetime.now()
    )
    new_data = stock_quote.filter(["Close"])
    new_dataset = new_data.values
    training_data_len = int(np.ceil(len(new_dataset) * 0.95))
    scaled_data = scaler.fit_transform(new_dataset)
    train_data = scaled_data[0 : int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print("funny business")

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    test_data = scaled_data[training_data_len - 60 :, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = new_dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot the data
    train = new_data[:training_data_len]
    valid = new_data[training_data_len:]
    valid["Predictions"] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 6))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.plot(train["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", "Val", "Predictions"], loc="lower right")
    plt.savefig(f"static/{predictStock}.png")

    # Now to do a specific prediction
    last_60_days = new_data[-60:].values
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


if __name__ == "__main__":
    print(getPrediction("AAPL"))
