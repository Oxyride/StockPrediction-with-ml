import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.optimizers import Adam


def create_dataset(path):
    global features_col
    features_col = []
    data = pd.read_csv(path)
    data = data.drop('Date', axis=1)
    cols = data.columns

    unscaled_y = data['Open']

    # 最大値を1, 最小値を０に変換
    data_normaliser = MinMaxScaler()
    data = data_normaliser.fit_transform(data)
    data = pd.DataFrame(data, columns=cols)
    data['raw_Open'] = unscaled_y

    def calc_ema(data, time_period):
        global features_col
        k = 2 / (1 - time_period)
        data['SMA'] = data['Close'].rolling(time_period).mean()
        # data['EMA'] = data['Close'] * k + data['SMA'].shift(1) * (1 - k)
        data['lag_1_SMA'] = data['SMA'].shift(1)
        # data['log_1_EMA'] = data['EMA'].shift(1)
        features_col.append('lag_1_SMA')
        # features_col.append('lag_1_EMA')

    def histories_nomalized(data):
        global features_col
        for i in range(1, history_points + 1):
            for c in cols:
                col = 'lag_{}_{}'.format(i, c)
                data[col] = data[c].shift(i)
                features_col.append(col)

    # calc_ema(data, 10)
    histories_nomalized(data)
    data.dropna(inplace=True)

    y_normaliser = MinMaxScaler()
    y_normaliser.fit(data['raw_Open'].values.reshape(-1, 1))
    return data, y_normaliser


def train_test_dataset():
    X_train = data[features_col][:N]
    y_train = data['Open'][:N]
    X_test = data[features_col][N:]
    y_test = data['Open'][N:]

    X_train = np.expand_dims(X_train, 2)
    X_test = np.expand_dims(X_test, 2)
    return X_train, y_train, X_test, y_test


def create_model_and_train():
    model = Sequential()
    model.add(LSTM(HIDEEN_LAYER, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(HIDEEN_LAYER))
    model.add(Dense(1))
    adam = Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=VALIDATION_SPLIT)
    return model


def evaluate_model_and_plot_result():
    preds = model.predict(X_test)
    preds_unscaled = y_normaliser.inverse_transform(preds)

    y_test.reset_index(drop=True, inplace=True)
    y_test_unscaled = y_normaliser.inverse_transform(y_test.values.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(preds_unscaled, label='Predict')
    plt.plot(y_test_unscaled, label='Real')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    path = 'data/etf_4751.csv'
    history_points = 30
    data, y_normaliser = create_dataset(path)

    TRAIN_SIZE = 0.9
    N = int(data.shape[0] * TRAIN_SIZE)
    HIDEEN_LAYER = 50
    DROPOUT = 0.5
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.1

    X_train, y_train, X_test, y_test = train_test_dataset()
    model = create_model_and_train()
    evaluate_model_and_plot_result()
