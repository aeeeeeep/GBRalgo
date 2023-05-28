import argparse
import os
import pickle
import re

from sklearn import ensemble
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

from utils import *


class GBR():
    def __init__(self, test_size=0.1, seed=42, path='resource/features.xlsx', feature='', grid=False, **params) -> None:
        self.test_size = test_size
        self.seed = seed
        self.params = params
        self.path = path
        self.feature = feature
        if grid:
            self.reg = ensemble.GradientBoostingRegressor()
        else:
            self.reg = ensemble.GradientBoostingRegressor(**self.params)
        self.mm_x = preprocessing.StandardScaler()
        self.mm_y = preprocessing.StandardScaler()

    def get_data(self):
        df = pd.read_excel(self.path)
        df.iloc[:, 1:] = df.iloc[:, 1:].clip(lower=0)
        df1 = df.loc[:,
              ['time', '1_PM2.5', '1_PM10', '1_NO2', '1_temperature', '1_humidity', 'CO mg/m³', 'NO₂ μg/m³',
               'PM₁₀ μg/m³',
               'PM₂.₅ μg/m³', 'SO₂ μg/m³', 'O₃ μg/m³', 'temperature', 'humidity', 'pressure hPa', 'wind velocity m/s',
               'wind direction']]
        df2 = df.loc[:,
              ['time', '2_PM2.5', '2_PM10', '2_NO2', '2_temperature', '2_humidity', 'CO mg/m³', 'NO₂ μg/m³',
               'PM₁₀ μg/m³',
               'PM₂.₅ μg/m³', 'SO₂ μg/m³', 'O₃ μg/m³', 'temperature', 'humidity', 'pressure hPa', 'wind velocity m/s',
               'wind direction']]
        df2.rename(
            columns={'2_PM2.5': '1_PM2.5', '2_PM10': '1_PM10', '2_NO2': '1_NO2', '2_temperature': '1_temperature',
                     '2_humidity': '1_humidity'}, inplace=True)

        data = pd.concat((df1, df2), axis=0)
        # data = self.cls_data(data)
        data = hour2sincos(data)
        # print(data.head())
        return data

    def cls_data(self, data):
        temperature_k = [10,20,20] # temperature
        pm25_k = [50, 30, 20] # PM2.5
        pm10_k = [50, 40, 40]# PM10
        no2_k = [60, 50, 50] # NO2
        humidity_k = [40, 30, 30] # humidity

        # 分类 1：可分辨, 0：异常
        # temperature
        data['clas_temperaturea'] = data['1_temperature'] - data['temperature']
        data['clas_temperatureb'] = data['clas_temperaturea'] - data['clas_temperaturea'].shift(-1)
        data['clas_temperaturec'] = data['1_temperature'] - data['1_temperature'].shift(-1)
        data['temperature_k'] = np.array(np.where((
                    data['clas_temperaturea'] < -temperature_k[0]) | (data['clas_temperaturea'] > temperature_k[0]) | (
                    data['clas_temperatureb'] < -temperature_k[1]) | (data['clas_temperatureb'] > temperature_k[1]) | (
                    data['clas_temperaturec'] < -temperature_k[2]) | (data['clas_temperaturec'] > temperature_k[2]), '0', '1'),dtype=bool)
        # PM2.5
        data['clas_pm2.5a'] = data['1_PM2.5'] - data['PM₂.₅ μg/m³']
        data['clas_pm2.5b'] = data['clas_pm2.5a'] - data['clas_pm2.5a'].shift(-1)
        data['clas_pm2.5c'] = data['1_PM2.5'] - data['1_PM2.5'].shift(-1)
        data['pm2.5_k'] = np.array(np.where((
                    data['clas_pm2.5a'] < -pm25_k[0]) | (data['clas_pm2.5a'] > pm25_k[0]) | (
                    data['clas_pm2.5b'] < -pm25_k[1]) | (data['clas_pm2.5b'] > pm25_k[1]) | (
                    data['clas_pm2.5c'] < -pm25_k[2]) | (data['clas_pm2.5c'] > pm25_k[2]), '0', '1'),dtype=bool)
        # PM10
        data['clas_pm10a'] = data['1_PM10'] - data['PM₁₀ μg/m³']
        data['clas_pm10b'] = data['clas_pm10a'] - data['clas_pm10a'].shift(-1)
        data['clas_pm10c'] = data['1_PM10'] - data['1_PM10'].shift(-1)
        data['pm10_k'] = np.array(np.where(((
                data['clas_pm10a'] < -pm10_k[0]) | (data['clas_pm10a'] > pm10_k[0]) | (
                data['clas_pm10b'] < -pm10_k[1]) | (data['clas_pm10b'] > pm10_k[1]) | (
                data['clas_pm10c'] < -pm10_k[2]) | (data['clas_pm10c'] > pm10_k[2])), '0', '1'),dtype=bool)
        # NO2
        data['clas_no2a'] = data['1_NO2'] - data['NO₂ μg/m³']
        data['clas_no2b'] = data['clas_no2a'] - data['clas_no2a'].shift(-1)
        data['clas_no2c'] = data['1_NO2'] - data['1_NO2'].shift(-1)
        data['no2_k'] = np.array(np.where((
                data['clas_no2a'] < -no2_k[0]) | (data['clas_no2a'] > no2_k[0]) | (
                data['clas_no2b'] < -no2_k[1]) | (data['clas_no2b'] > no2_k[1]) | (
                data['clas_no2c'] < -no2_k[2]) | (data['clas_no2c'] > no2_k[2]), '0', '1'), dtype=bool)
        # humidity
        data['clas_humiditya'] = data['1_humidity'] - data['humidity']
        data['clas_humidityb'] = data['clas_humiditya'] - data['clas_humiditya'].shift(-1)
        data['clas_humidityc'] = data['1_humidity'] - data['1_humidity'].shift(-1)
        data['humidity_k'] = np.array(np.where((
                data['clas_humiditya'] < -humidity_k[0]) | (data['clas_humiditya'] > humidity_k[0]) | (
                data['clas_humidityb'] < -humidity_k[1]) | (data['clas_humidityb'] > humidity_k[1]) | (
                data['clas_humidityc'] < -humidity_k[2]) | (data['clas_humidityc'] > humidity_k[2]), '0', '1'),dtype=bool)

        if self.feature == 'PM2.5':
            data.drop(data[data['pm2.5_k'] == False].index, inplace=True)
        if self.feature == 'PM10':
            data.drop(data[data['pm10_k'] == False].index, inplace=True)
        if self.feature == 'NO2':
            data.drop(data[data['no2_k'] == False].index, inplace=True)
        if self.feature == 'temperature':
            data.drop(data[data['temperature_k'] == False].index, inplace=True)
        if self.feature == 'humidity':
            data.drop(data[data['humidity_k'] == False].index, inplace=True)
        return data

    def get_features(self, data):
        print("predict feature: {}, length: {}".format(self.feature, len(data)))
        if self.feature == 'PM2.5':
            fearture_name = ['humidity', 'temperature', 'O₃ μg/m³', 'CO mg/m³', 'SO₂ μg/m³', 'wind velocity m/s',
                             'wind direction', 'pressure hPa', '1_PM10', '1_PM2.5', 'sin_time','cos_time']
            # y = (data['PM₂.₅ μg/m³']-data['1_PM2.5']).values
            y = data['PM₂.₅ μg/m³'].values
        elif self.feature == 'PM10':
            fearture_name = ['humidity', 'temperature', 'NO₂ μg/m³', 'CO mg/m³', 'SO₂ μg/m³', 'wind velocity m/s',
                             'wind direction', 'pressure hPa', '1_PM10', 'sin_time','cos_time']
            # y = (data['PM₁₀ μg/m³']-data['1_PM10']).values
            y = data['PM₁₀ μg/m³'].values
        elif self.feature == 'NO2':
            fearture_name = ['humidity', 'temperature', 'CO mg/m³', 'SO₂ μg/m³', 'O₃ μg/m³', 'wind velocity m/s',
                             'wind direction', 'pressure hPa', '1_NO2', 'sin_time', 'cos_time']
            # y = (data['NO₂ μg/m³']-data['1_NO2']).values
            y = data['NO₂ μg/m³'].values
        elif self.feature == 'temperature':
            fearture_name = ['humidity', '1_temperature', 'wind velocity m/s', 'wind direction', 'pressure hPa',
                             'O₃ μg/m³', 'NO₂ μg/m³', 'sin_time', 'cos_time']
            # y = (data['temperature']-data['1_temperature']).values
            y = data['temperature'].values
        elif self.feature == 'humidity':
            fearture_name = ['1_humidity', 'temperature', 'wind velocity m/s', 'wind direction', 'pressure hPa',
                             'PM₁₀ μg/m³', 'O₃ μg/m³', 'NO₂ μg/m³', 'sin_time']
            # y = (data['humidity']-data['1_humidity']).values
            y = data['humidity'].values
        else:
            raise Exception("Feature not exist!")
        x = data[fearture_name]
        """plot
        plt.figure(figsize=(10,3),dpi=300)
        plt.plot(y,linewidth=0.5)
        y = multiscale_analysis(y)
        plt.plot(y,linewidth=0.3, linestyle='--')
        plt.show()
        """

        x_scaled = self.mm_x.fit_transform(x)
        y_scaled = self.mm_y.fit_transform(y.reshape(-1, 1))

        x_train = x_scaled[:int(len(x_scaled)*(1-self.test_size))]
        x_test = x_scaled[int(len(x_scaled)*(1-self.test_size)):]
        y_train = y_scaled[:int(len(x_scaled)*(1-self.test_size))]
        y_test = y_scaled[int(len(x_scaled)*(1-self.test_size)):].ravel()

        # x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled.ravel(), test_size=self.test_size,
        #                                                     random_state=self.seed)
        return x_train, x_test, y_train, y_test

    def run(self, x_train, x_test, y_train, y_test):
        self.reg.fit(x_train, y_train)
        ytrain_pre = self.reg.predict(x_train)
        ytest_pre = self.reg.predict(x_test)

        test_score = np.zeros((self.params["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(self.reg.staged_predict(x_test)):
            test_score[i] = self.reg.loss_(y_test, y_pred)

        # result = permutation_importance(
        #     self.reg, x_test, y_test, n_repeats=15, random_state=12, n_jobs=1
        # )

        train_mse = mean_squared_error(y_train, self.reg.predict(x_train))
        test_mse = mean_squared_error(y_test, self.reg.predict(x_test))
        train_score = self.reg.score(x_train, y_train)
        test_score = self.reg.score(x_test, y_test)
        print("The mean squared error (MSE) on train set: {:.4f}".format(train_mse))
        print("The mean squared error (MSE) on test set: {:.4f}".format(test_mse))
        print("The R^2 on train set: {:.4f}".format(train_score))
        print("The R^2 on test set: {:.4f}".format(test_score))

        self.save_model(test_score)

    def search_grid(self, x_train, x_test, y_train, y_test):
        grid_search = GridSearchCV(self.reg, param_grid=self.params, cv=5)
        grid_search.fit(x_train, y_train)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", -grid_search.best_score_)

    def save_model(self, test_score):
        files = os.listdir('model')
        for file in files:
            if self.feature in file:
                score = re.findall(r'\d+\.\d{4}', file)[0]
                if test_score > float(score):
                    with open("model/{}_{:.4f}.pickle".format(self.feature, test_score), 'wb') as f:
                        pickle.dump(self.reg, f)
                return
        with open("model/{}_{:.4f}.pickle".format(self.feature, test_score), 'wb') as f:
            pickle.dump(self.reg, f)

    def load_model(self, feature):
        files = os.listdir('model')
        for file in files:
            if feature in file:
                with open("model/{}".format(file), 'rb') as f:
                    return pickle.load(f)
        raise Exception("Model not exist!")

    def pred(self, feature, x):
        model = self.load_model(feature)
        if feature != 'PM2.5' and feature != 'PM10' and feature != 'NO2' and feature != 'temperature' and feature != 'humidity':
            raise Exception("Feature not exist!")
            return
        x = self.mm_x.transform(x)
        y_pred = model.predict(x)
        y_pred = self.mm_y.inverse_transform(y_pred.reshape(-1, 1))
        return y_pred

    def get_params(feature):
        if feature == 'NO2':
            params = {'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 12, 'n_estimators': 1200}
        elif feature == 'humidity':
            params = {'learning_rate': 0.05, 'max_depth': 8, 'min_samples_split': 7, 'n_estimators': 900}
        elif feature == 'temperature':
            params = {'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 6, 'n_estimators': 900}
        elif feature == 'PM10':
            params = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 12, 'n_estimators': 1200}
        elif feature == 'PM2.5':
            params = {'learning_rate': 0.05, 'max_depth': 7, 'min_samples_split': 12, 'n_estimators': 1200}
        else:
            raise Exception("Feature not exist!")
        return params
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GBR model train & predict')
    parser.add_argument('--run', '-r', help='run train & pred feature', default='')
    parser.add_argument('--grid', '-g', help='search better grid', default='')
    parser.add_argument('--predict', '-p', help='predict feature', default='')
    parser.add_argument('--file', '-f', help='feature file', default='resource/features.xlsx')
    args = parser.parse_args()

    if args.run == 'NO2' or args.predict == 'NO2':
        params = {'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 12, 'n_estimators': 1200}
    elif args.run == 'humidity' or args.predict == 'humidity':
        params = {'learning_rate': 0.05, 'max_depth': 8, 'min_samples_split': 7, 'n_estimators': 900}
    elif args.run == 'temperature' or args.predict == 'temperature':
        params = {'learning_rate': 0.05, 'max_depth': 6, 'min_samples_split': 6, 'n_estimators': 900}
    elif args.run == 'PM10' or args.predict == 'PM10':
        params = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 12, 'n_estimators': 1200}
    elif args.run == 'PM2.5' or args.predict == 'PM2.5':
        params = {'learning_rate': 0.05, 'max_depth': 7, 'min_samples_split': 12, 'n_estimators': 1200}
    else:
        raise Exception("Feature not exist!")

    params_grid = {
        'n_estimators': [400, 600, 800, 1000, 1200, 1400, 1600, 1800],
        'max_depth': [5, 7, 9, 11, 13],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_split': [4, 8, 12, 16]
    }

    if args.run != '':
        gbr = GBR(test_size=0.1, seed=42, path=args.file, feature=args.run, **params)
        data = gbr.get_data()
        x_train, x_test, y_train, y_test = gbr.get_features(data)
        gbr.run(x_train, x_test, y_train, y_test)

    if args.predict != '':
        gbr = GBR(test_size=0.1, seed=42, path=args.file, feature=args.predict, **params)
        data = gbr.get_data()
        # x_train, x_test, y_train, y_test = gbr.get_features(data)
        # gbr.load_model(args.predict)
        # y_pred = gbr.pred(x_test)
        # print(mean_squared_error(y_test, y_pred))

    if args.grid != '':
        gbr = GBR(test_size=0.0001, seed=42, path=args.file, feature=args.grid, grid=True, **params_grid)
        data = gbr.get_data()
        x_train, x_test, y_train, y_test = gbr.get_features(data)
        gbr.search_grid(x_train, x_test, y_train, y_test)
x=[]
feature=''
y=GBR.pred(feature,x)