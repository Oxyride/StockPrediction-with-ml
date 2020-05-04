import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.optimizers import Adam

history = 20

# データセット作成
def make_dataset(path):
    data = pd.read_csv('data/etf_1321.csv')
    data = data.drop('Date', axis=1)
    unscaled_y = data['Close'][history + 1:].values.reshape(-1, 1)

    # 最大値を1, 最小値を０に変換
    data_normaliser = MinMaxScaler()
    data_normaliser.fit(data)
    data = data_normaliser.transform(data)

    # history日前から前日までのデータを特徴量にする
    data = np.array([data[i - history:i] for i in range(history, len(data))])

    y_normaliser = MinMaxScaler()
    y_normaliser.fit(unscaled_y)
    scaled_y = y_normaliser.transform(unscaled_y)
    return data, y_normaliser, scaled_y


path = 'data/etf_1321.csv'
data, y_normaliser, scaled_y = make_dataset(path)

# 各パラメータ
TRAIN_SIZE = 0.9
N = int(data.shape[0] * TRAIN_SIZE)
HIDEEN_LAYER = 50
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.1

# トレイン、テストセットに分割
X_train = data[:N]
y_train = scaled_y[:N]
X_test = data[N:]
y_test = scaled_y[N:]


# LSTMモデル作成
model = Sequential()
model.add(LSTM(HIDEEN_LAYER, return_sequences=True, input_shape=(X_train.shape[1], 6)))
model.add(Dropout(DROPOUT))
model.add(LSTM(HIDEEN_LAYER))
model.add(Dropout(DROPOUT))
model.add(Dense(1))
adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=VALIDATION_SPLIT)


preds = model.predict(X_test)
preds_unscaled = y_normaliser.inverse_transform(preds)
y_test_unscaled = y_normaliser.inverse_transform(y_test)

# 予測結果　表示
plt.figure(figsize=(20, 5))
plt.plot(preds_unscaled, label='Predict')
plt.plot(y_test_unscaled, label='Real')
plt.legend()
