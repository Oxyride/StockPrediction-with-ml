import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import plt
from keras.models import Model
from keras.layers import LSTM, Dropout, Dense, Activation, Input
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

history = 23


def make_dataset(path):
    data = pd.read_csv(path)
    data = data.drop('Date', axis=1)
    unscaled_y = data['Close'][history + 1:].values.reshape(-1, 1)

    data_normaliser = MinMaxScaler()
    data_normaliser.fit(data)
    data = data_normaliser.transform(data)

    def calc_sma(time_period):
        return np.array([[data[i:i + time_period][0].mean()] for i in range(len(data) - history)])

    def make_lag_data():
        return np.array([data[i - history:i] for i in range(history, len(data))])

    y_normaliser = MinMaxScaler()
    y_normaliser.fit(unscaled_y)
    indicater = calc_sma(5)
    data = make_lag_data()
    scaled_y = y_normaliser.transform(unscaled_y)
    return data, indicater, scaled_y, y_normaliser


path = 'data/etf_1321.csv'
data, indicater, scaled_y, y_normaliser = make_dataset(path)

TRAIN_SIZE = 0.9
N = int(data.shape[0] * TRAIN_SIZE)
HIDEEN_LAYER = 50
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.1

X_train = data[:N]
indicater_train = indicater[:N]
y_train = scaled_y[:N]
X_test = data[N:]
indicater_test = indicater[N:]
y_test = scaled_y[N:]

lstm_input = Input(shape=(history, 6))
dense_input = Input(shape=(indicater.shape[1],))

x = LSTM(HIDEEN_LAYER)(lstm_input)
x = Dropout(DROPOUT)(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

y = Dense(20)(dense_input)
y = Activation('relu')(y)
y = Dropout(DROPOUT)(y)
indicater_branch = Model(inputs=dense_input, outputs=y)

combined = concatenate([lstm_branch.output, indicater_branch.output])
z = Dense(64, activation='sigmoid')(combined)
z = Dense(1, activation='linear')(z)

model = Model(inputs=[lstm_branch.input, indicater_branch.input], outputs=z)
plot_model(model, to_file='stock_prediction_LSTM_v2_model.png', show_shapes=True)

adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=[X_train, indicater_train], y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
          validation_split=VALIDATION_SPLIT)


preds = model.predict([X_test, indicater_test])
preds_unscaled = y_normaliser.inverse_transform(preds)
y_test_unscaled = y_normaliser.inverse_transform(y_test)


plt.figure(figsize=(20, 5))
plt.plot(preds_unscaled, label='Predict')
plt.plot(y_test_unscaled, label='Real')
plt.legend()
plt.savefig('stock_prediction_LSTM_v2_result.png')
