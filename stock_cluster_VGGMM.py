import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import config
from sklearn import preprocessing, mixture


# データを整頓する
def shape_data(data, start=dt.datetime(2015, 1, 1)):
    col = data.columns[-1]
    data = data[['Date', col]].copy()
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    return data[data.index > start]


# 全データを結合
def make_data():
    etf_list = config.ETFLIST[:100]
    for i, etf in enumerate(etf_list):
        path = 'data/etf_' + str(etf) + '.csv'
        etf_file = pd.read_csv(path)
        etf_data = shape_data(etf_file)
        if i == 0:
            data = etf_data.copy()
        else:
            data = pd.concat([data, etf_data['etf' + str(etf)]], axis=1)
    return data


# データを相関係数の降順で並べ換える
def make_ranking(target_data):
    data = target_data.copy()
    ranking = data.corrwith(data[target])
    ranking.sort_values(ascending=False, inplace=True)
    return ranking


# lag特徴量を作成
def create_lags(data):
    global cols
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)


# ビンに対応する位置を特徴量に追加
def create_bins(data, bins=[0]):
    global cols_bin
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)


# 　NaNを中央値で補完
def data_fill_na(pre_data):
    return pre_data.fillna(pre_data.median())


# vbgmmで似た動きの株をクラスタリング
def vbgmm_cluster(pri_data, n_components):
    data = pri_data.T
    sc = preprocessing.StandardScaler()
    sc.fit(data)
    data_norm = sc.transform(data)
    vbgm = mixture.BayesianGaussianMixture(n_components=n_components)
    vbgm = vbgm.fit(data_norm)
    labels = vbgm.predict(data_norm)
    data['class'] = labels
    data = data[data['class'] == data['class'][target]]
    return data.T


# 同じクラスの株価をプロット
def plot_result(pri_data, cluster_data):
    plt.figure(figsize=(25, 10))
    for col in cluster_data.columns:
        plt.plot(pri_data[col])
    plt.savefig('cluster_' + target + '.png')
    plt.show()


def run_all():
    data = make_data()
    data_fill = data_fill_na(data)
    cluster_data = vbgmm_cluster(data_fill, 10)
    plot_result(data, cluster_data)


if __name__ == '__main__':
    start = dt.datetime(2018, 5, 1)
    target = 'etf1301'
    lags = 5
    run_all()
