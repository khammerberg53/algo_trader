# essentially a copy of LSTM from different repository on my profile, but modified slightly to be used witha algo trader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import yfinance as yf

### Stock price predictor
### To visualize training process w/ TensorBoard run cmd in terminal window: tensorboard --logdir logs/fit


# Reproducability
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

### Set parameters

N_STEPS = 100
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
PERIOD = '1d'
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
INTERVAL = '1m'
# Lookup step, 1 is the next day
LOOKUP_STEP = 1
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.3
# features to use
FEATURE_COLUMNS = ["Close", "Volume", "Open", "High", "Low"]
# date now
date_now = time.strftime("%Y-%m-%d")

# > model parameters <
N_LAYERS = 5
#Type of model
CELL = LSTM
# Number of neurons
UNITS = 256
# Dropout rate
DROPOUT = 0.3
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

# > training parameters <
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 40
ticker = "AAPL"


#units = neurons
def load_data(ticker, period, interval, n_steps=200, scale=True, shuffle=True, lookup_step=30, test_size=.2,
              feature_columns=['Close', 'Volume', 'Open', 'High', 'Low']):
    '''
    :param ticker: Ticker you want to load, dtype: str
    :param period: Time period you want data from, dtype: str(options in program)
    :param interval: Interval for data, dtype:str
    :param n_steps: Past sequence length used to predict, default = 50, dtype: int
    :param scale: Whether to scale data b/w 0 and 1, default = True, dtype: Bool
    :param shuffle: Whether to shuffle data, default = True, dtyper: Bool
    :param lookup_step: Future lookup step to predict, default = 1(next day), dtype:int
    :param test_size: ratio for test data, default is .2 (20% test data), dtype: float
    :param feature_columns: list of features fed into the model, default is OHLCV, dtype: list
    :return:
    '''
    df = yf.download(tickers=ticker, period=period, interval=interval,
                     group_by='ticker',
                     # adjust all OHLC automatically
                     auto_adjust=True, prepost=True, threads=True, proxy=None)

    result = {}
    result['df'] = df.copy()
    ### preview data frame before preprocessing
    print(df)

    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                               test_size=test_size, shuffle=shuffle)
    # return the result
    return result

def create_model(sequence_length, units=256, cell=LSTM, n_layers=3, dropout=0.3,
                loss="mean_absolute_error", optimizer="adam", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

#save model
model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"

# folders that store results
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

data = load_data(ticker, PERIOD, INTERVAL, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
# save the dataframe
data["df"].to_csv()
# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
model.save(os.path.join("results", model_name) + ".h5")

# >> Testing the Model <<

data = load_data(ticker, PERIOD, INTERVAL, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)
# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = data["column_scaler"]["Close"].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)

def predict(model, data):
    last_sequence = data["last_sequence"][-N_STEPS:]
    column_scaler = data["column_scaler"]
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["Close"].inverse_transform(prediction)[0][0]
    return predicted_price

tickerData = yf.Ticker(f'{ticker}')
tickerDf = tickerData.history(period='7d')
tickerDf

# predict the future price
future_price = predict(model, data)
print(f"Recent {ticker} QQQ pricing:")
print(f"{tickerDf}")
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")

def plot_graph(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))
    # currently last 200 days
    plt.plot(y_test[-100:], c='b')
    plt.plot(y_pred[-100:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    ### added plot title
    plt.title(f'{ticker}')
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

plot_graph(model, data)

### inverse transformation back to price from normalized values and then converted to boolean and calculated
def accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)

def accuracy2(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))
    return r2_score(y_test, y_pred)

print(str(LOOKUP_STEP) + ":", "Accuracy Score:", accuracy(model, data))

print(str(LOOKUP_STEP) + ":", "R2:", accuracy2(model, data))
