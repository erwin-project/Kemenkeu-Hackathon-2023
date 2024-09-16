import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from utils import visualization as vs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Conv1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def all_model(kind_model):
    model = {'Linear Regression': LinearRegression(),
             'Logistic Regression': LogisticRegression(),
             'Bayesian Ridge Regression': BayesianRidge(),
             'SVR': SVR(kernel='rbf'),
             'Decision Tree Regression': DecisionTreeRegressor(),
             'CNN': Sequential(),
             'LSTM': Sequential()}

    return model[kind_model]


def linear_regression(data1, data2):
    x = np.array(data1).reshape(-1, 1)
    y = np.array(data2).reshape(-1, 1)

    regr = LinearRegression()
    regr.fit(x, y)

    score = regr.score(x, y)
    y_pred = regr.predict(x)

    return y_pred, score


def model_dbscan(data_ml, target):
    data_ml.drop(['Provinsi',
                  target], inplace=True, axis=1)

    # Declaring Model
    dbscan = DBSCAN()

    # Fitting
    dbscan.fit(data_ml)

    # Transforming Using PCA
    pca = PCA(n_components=3).fit(data_ml.values)
    pca_2d = pca.transform(data_ml.values)

    fig, ax = plt.subplots(1, figsize=(10, 8))

    # Plot based on Class
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif dbscan.labels_[i] == 2:
            ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y', marker='-')
        elif dbscan.labels_[i] == -1:
            ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    # ax.legend()
    ax.set_title('DBSCAN finds 3 clusters and Noise')

    return fig, ax, dbscan.labels_


def supervised_learning(kind_model, scaler, data_ml, data_ml_proj, target, years, proj_years):
    X = data_ml.drop(['Provinsi',
                      target], axis=1)

    y = data_ml[target]

    X_proj = data_ml_proj.drop(['Provinsi',
                                target], axis=1)

    y_proj = data_ml_proj[target]

    # Dropping any rows with Nan values
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)

    # Splitting the data into training and testing data
    build_ml = all_model(kind_model)

    build_ml.fit(X_train, y_train)
    score = build_ml.score(X_test, y_test)

    data_predict = build_ml.predict(X_proj.values)

    data_ml_true = pd.DataFrame({'Provinsi': data_ml['Provinsi'].values,
                                 years: data_ml[target].values,
                                 proj_years: data_predict})

    chart_datas = pd.melt(data_ml_true,
                          id_vars=["Provinsi"])

    title1 = "Efficiency Projection Each Province in " + str(years)
    title2 = "Efficiency Projection Each Province in " + str(proj_years)

    chart1 = vs.get_bar_vertical(chart_datas[chart_datas['variable'] == years],
                                 "Provinsi",
                                 "value",
                                 "variable",
                                 "Province",
                                 "Efficiency Value",
                                 title1)

    chart2 = vs.get_bar_vertical(chart_datas[chart_datas['variable'] == proj_years],
                                 "Provinsi",
                                 "value",
                                 "variable",
                                 "Province",
                                 "Efficiency Value",
                                 title2)

    return chart1, chart2, score, data_ml_true


def unsupervised_learning(kind_model, scaler, data_ml, data_ml_proj, target, years, proj_years):
    X = data_ml.drop(['Provinsi',
                      target], axis=1)

    y = data_ml[target]

    X_proj = data_ml_proj.drop(['Provinsi',
                                target], axis=1)

    y_proj = data_ml_proj[target]

    # Dropping any rows with Nan values
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    model = Sequential()

    if kind_model == "LSTM":
        # create and fit the LSTM network
        model.add(LSTM(4, input_shape=(1, X_train.shape[1])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
        model.fit(trainX, y_train, epochs=100, batch_size=1, verbose=2)

    elif kind_model == "ANN":
        model.add(Dense(128,
                        input_dim=X_train.shape[1]))
        model.add(Dense(64,
                        input_dim=X_train.shape[1]))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1,
                        activation='softmax'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
        model.fit(trainX, y_train, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([y_train])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([y_test])

    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    X_proj = np.reshape(X_proj.values, (X_proj.shape[0], 1, X_proj.shape[1]))

    data_predict = model.predict(X_proj)

    datas_predict = []

    for data in data_predict:
        datas_predict.append(data[0])

    data_ml_true = pd.DataFrame({'Provinsi': data_ml['Provinsi'].values,
                                 years: data_ml[target].values,
                                 proj_years: datas_predict})

    chart_datas = pd.melt(data_ml_true,
                          id_vars=["Provinsi"])

    title1 = "Efficiency Projection Each Province in " + str(years)
    title2 = "Efficiency Projection Each Province in " + str(proj_years)

    chart1 = vs.get_bar_vertical(chart_datas[chart_datas['variable'] == years],
                                 "Provinsi",
                                 "value",
                                 "variable",
                                 "Province",
                                 "Efficiency Value",
                                 title1)

    chart2 = vs.get_bar_vertical(chart_datas[chart_datas['variable'] == proj_years],
                                 "Provinsi",
                                 "value",
                                 "variable",
                                 "Province",
                                 "Efficiency Value",
                                 title2)

    return chart1, chart2, trainScore, data_ml_true


def convert_data_efficiency(path_data, output):
    target_ALL = ['Pendapatan_per_Kapita_per_bulan',
                  'Anggaran_Kesehatan_per_Kapita',
                  'Anggaran_Kesehatan',
                  'Cakupan_JKN',
                  'Pertumbuhan_Cakupan_JKN',
                  'TTRS_per_1000',
                  'Pertumbuhan_TTRS_per_1000',
                  'Tenaga_Medis_per_10k_Populasi',
                  'Pertumbuhan_Tenaga_Medis_per10k',
                  'AHH',
                  'Pertumbuhan_AHH']

    output_ALL = [output]

    data_prov = pd.read_excel(path_data,
                              sheet_name=target_ALL[0])
    data_true_1 = pd.DataFrame({'Provinsi': data_prov['Provinsi'].values})
    data_true_2 = pd.DataFrame({'Provinsi': data_prov['Provinsi'].values})

    for col in target_ALL:
        dataset = pd.read_excel(path_data,
                                sheet_name=col)

        data_true_1[col] = dataset[2019]
        data_true_2[col] = dataset[2020]

    data_true_target = pd.concat([data_true_1, data_true_2])

    data_prov = pd.read_excel(path_data,
                              sheet_name=output_ALL[0])
    data_true_1 = pd.DataFrame({'Provinsi': data_prov['Provinsi'].values})
    data_true_2 = pd.DataFrame({'Provinsi': data_prov['Provinsi'].values})

    for col in output_ALL:
        dataset = pd.read_excel(path_data,
                                sheet_name=col)

        data_true_1[col] = dataset[2019]
        data_true_2[col] = dataset[2020]

    data_true_output = pd.concat([data_true_1, data_true_2])

    pertumbuhan = ['Pertumbuhan_Cakupan_JKN',
                   'Pertumbuhan_TTRS_per_1000',
                   'Pertumbuhan_Tenaga_Medis_per10k',
                   'Pertumbuhan_AHH']

    data_prov = pd.read_excel(path_data,
                              sheet_name=pertumbuhan[0])
    data_true_proj = pd.DataFrame({'Provinsi': data_prov['Provinsi'].values})

    for col in target_ALL:
        dataset = pd.read_excel(path_data,
                                sheet_name=col)

        if col in pertumbuhan:
            data_true_proj[col], score = linear_regression(dataset[2019].values,
                                                           dataset[2020].values)
        else:
            data_true_proj[col] = dataset[2021]

    X = data_true_target.drop(['Provinsi'], axis=1)
    Y = data_true_output.drop(['Provinsi'], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2)
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_proj = data_true_proj.drop(['Provinsi'], axis=1)

    # for data_col in data_proj.columns:
    #     data_proj[data_col] = scaler.fit_transform(data_proj[data_col].values.reshape(-1, 1))

    build_ml = LinearRegression()

    build_ml.fit(X_train, Y_train)
    score = build_ml.score(X_test, Y_test)
    data_predict = build_ml.predict(data_proj.values)

    val_predict = []

    for val in data_predict:
        val_predict.append(val[0])

    data_ml = pd.DataFrame({'Provinsi': data_prov['Provinsi'].values,
                            2021: val_predict})

    return data_ml, score

