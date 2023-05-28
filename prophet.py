import pickle
from abc import abstractmethod
from warnings import simplefilter

import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics

simplefilter(action="ignore", category=FutureWarning)


class ProphetTrain():
    def __init__(self, name=None):
        self.name = name
        self.data: pd.DataFrame = None
        self.params = None
        self.mse = np.inf
        self.model = None
        self.ratio = 0.9
        self.grid_search_params_path = None

    @abstractmethod
    def _load_data(self):
        """
        加载训练及测试数据
        :param rule: DataFrame.resample 中的rule
        :return: 训练及测试数据集，类型是pd.DataFrame
        """
        micro1 = pd.read_excel('./resource/micro1.xlsx')
        micro2 = pd.read_excel('./resource/micro2.xlsx')
        standard = pd.read_excel('./resource/standard.xlsx')
        micro1.iloc[:, 2:] = micro1.iloc[:, 2:].clip(lower=0)
        micro2.iloc[:, 2:] = micro2.iloc[:, 2:].clip(lower=0)
        standard.iloc[:, 2:] = standard.iloc[:, 2:].clip(lower=0)
        micro1 = micro1.iloc[:, 1:]
        micro2 = micro2.iloc[:, 1:]
        standard = standard.iloc[:, 1:]
        micro1_diff = pd.DataFrame()
        micro1_diff['PM2.5'] = micro1['PM2.5'] - standard['PM₂.₅ μg/m³']
        micro1_diff['PM10'] = micro1['PM10'] - standard['PM₁₀ μg/m³']
        micro1_diff['NO2'] = micro1['NO2'] - standard['NO₂ μg/m³']
        micro1_diff['temperature'] = micro1['temperature'] - standard['temperature']
        micro1_diff['humidity'] = micro1['humidity'] - standard['humidity']
        micro1['time'] = pd.to_datetime(micro1['time'])  # 将日期转换为Pandas的日期格式
        micro2['time'] = pd.to_datetime(micro2['time'])  # 将日期转换为Pandas的日期格式
        standard['time'] = pd.to_datetime(standard['time'])  # 将日期转换为Pandas的日期格式
        data = pd.concat([standard, micro1_diff['PM10']], axis=1)
        data = data.rename(columns={'time': 'ds', 'PM10': 'y'})  # 将列名改为Prophet所需的格式
        return data

    @property
    def data_size(self):
        if self.data is None:
            self.data = self._load_data()
        return self.data.shape[0] if self.data is not None else 0

    def _cv_run(self):
        if self.data_size < 14:
            raise Exception("数据量不足，请保证数据航速大于14条")
        if self.params is None:
            self.model = Prophet()
        else:
            print(self.params)
            self.model = Prophet(
                seasonality_mode=self.params['seasonality_mode'],
                changepoint_range=self.params['changepoint_range'],
                seasonality_prior_scale=self.params['seasonality_prior_scale'],
                holidays_prior_scale=self.params['holidays_prior_scale']
            )
        train_length = int(len(self.data) * self.ratio)
        train = self.data[:train_length]
        test = self.data[train_length:]
        self.model.fit(train)
        cv_result = cross_validation(self.model, f'{len(test)}h',
                                     f'{len(test)}h')
        return performance_metrics(cv_result, metrics=['mse'])['mse'][0]

    def run(self, show: int = 0, retrain=False):
        """
        根据当前参数生成模型
        :param retrain: 是否根据当前参数重新生成模型
        :param show:
        0:  不保存图片及预测结果 也 不展示图片
        1： 展示图片
        2： 保存图片及预测结果
        3： 保存图片及预测结果 也 展示图片
        :return:
        """
        if self.data_size < 14:
            raise Exception("数据量不足，请保证数据航速大于14条")
        if retrain or self.model is None:
            self.model = Prophet(self.params)

        self.model.add_regressor('CO mg/m³')
        self.model.add_regressor('NO₂ μg/m³')
        self.model.add_regressor('PM₁₀ μg/m³')
        self.model.add_regressor('PM₂.₅ μg/m³')
        self.model.add_regressor('SO₂ μg/m³')
        self.model.add_regressor('O₃ μg/m³')
        self.model.add_regressor('temperature')
        self.model.add_regressor('humidity')
        self.model.add_regressor('pressure hPa')
        self.model.add_regressor('wind velocity m/s')
        self.model.add_regressor('wind direction')

        train_length = int(len(self.data) * self.ratio)  # 计算训练数据集的长度，即前 80%
        train = self.data[:train_length]  # 选择前 80% 作为训练数据集
        test = self.data[train_length:]  # 选择后 20% 作为测试数据集
        self.model.fit(train)
        future = self.model.make_future_dataframe(periods=len(test), freq='24H')
        future['CO mg/m³'] = self.data['CO mg/m³'].values
        future['NO₂ μg/m³'] = self.data['NO₂ μg/m³'].values
        future['PM₁₀ μg/m³'] = self.data['PM₁₀ μg/m³'].values
        future['PM₂.₅ μg/m³'] = self.data['PM₂.₅ μg/m³'].values
        future['SO₂ μg/m³'] = self.data['SO₂ μg/m³'].values
        future['O₃ μg/m³'] = self.data['O₃ μg/m³'].values
        future['temperature'] = self.data['temperature'].values
        future['humidity'] = self.data['humidity'].values
        future['pressure hPa'] = self.data['pressure hPa'].values
        future['wind velocity m/s'] = self.data['wind velocity m/s'].values
        future['wind direction'] = self.data['wind direction'].values
        forecast = self.model.predict(future)
        if show & 0b01:
            self.model.plot(forecast).show()  # 绘制预测效果图
            self.model.plot_components(forecast).show()  # 绘制成分趋势图
        if show & 0b10:
            y = forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].iloc[-len(test):]
            y.to_csv(f'csv/{self.name}.csv', index=False)
            self.model.plot(forecast).savefig(f"img/{self.name}-scatter.png")  # 绘制预测效果图
            self.model.plot_components(forecast).savefig(f"img/{self.name}-trend.png")  # 绘制成分趋势图
        mape_score = np.abs(1 - forecast['yhat'].iloc[:self.data.shape[0]] / self.data['y'].values)
        return np.quantile(mape_score, self.ratio)

    @property
    def get_predict_df(self):
        train_length = int(len(self.data) * self.ratio)  # 计算训练数据集的长度，即前 80%
        test = self.data[train_length:]  # 选择后 20% 作为测试数据集
        future = self.model.make_future_dataframe(periods=len(test), freq='24H')
        future['CO mg/m³'] = self.data['CO mg/m³'].values
        future['NO₂ μg/m³'] = self.data['NO₂ μg/m³'].values
        future['PM₁₀ μg/m³'] = self.data['PM₁₀ μg/m³'].values
        future['PM₂.₅ μg/m³'] = self.data['PM₂.₅ μg/m³'].values
        future['SO₂ μg/m³'] = self.data['SO₂ μg/m³'].values
        future['O₃ μg/m³'] = self.data['O₃ μg/m³'].values
        future['temperature'] = self.data['temperature'].values
        future['humidity'] = self.data['humidity'].values
        future['pressure hPa'] = self.data['pressure hPa'].values
        future['wind velocity m/s'] = self.data['wind velocity m/s'].values
        future['wind direction'] = self.data['wind direction'].values
        forecast = self.model.predict(future)
        return forecast

    def grid_search(self, use_cv=True, save_result=True):
        """
        结合cv进行网格寻参，cv方式网格寻参很慢，一般建议先使用非网格方式，待参数调整完毕再使用cv验证。
        :param save_result:
        :return:
        """
        changepoint_range = [i / 10 for i in range(3, 10)]
        seasonality_mode = ['additive', 'multiplicative']
        seasonality_prior_scale = [0.05, 0.1, 0.5, 1, 5, 10, 15]
        holidays_prior_scale = [0.05, 0.1, 0.5, 1, 5, 10, 15]
        for sm in seasonality_mode:
            for cp in changepoint_range:
                for sp in seasonality_prior_scale:
                    for hp in holidays_prior_scale:
                        params = {"seasonality_mode": sm, "changepoint_range": cp, "seasonality_prior_scale": sp,
                                  "holidays_prior_scale": hp, "growth": "flat", }
                        score = self._cv_run() if use_cv else self.run()
                        if self.mse > score:
                            self.mse = score
                            self.params = params
        if save_result:
            train_length = int(len(self.data) * self.ratio)  # 计算训练数据集的长度，即前 80%
            test = self.data[train_length:]  # 选择后 20% 作为测试数据集
            future = self.model.make_future_dataframe(periods=len(test), freq='24H')
            future['CO mg/m³'] = self.data['CO mg/m³'].values
            future['NO₂ μg/m³'] = self.data['NO₂ μg/m³'].values
            future['PM₁₀ μg/m³'] = self.data['PM₁₀ μg/m³'].values
            future['PM₂.₅ μg/m³'] = self.data['PM₂.₅ μg/m³'].values
            future['SO₂ μg/m³'] = self.data['SO₂ μg/m³'].values
            future['O₃ μg/m³'] = self.data['O₃ μg/m³'].values
            future['temperature'] = self.data['temperature'].values
            future['humidity'] = self.data['humidity'].values
            future['pressure hPa'] = self.data['pressure hPa'].values
            future['wind velocity m/s'] = self.data['wind velocity m/s'].values
            future['wind direction'] = self.data['wind direction'].values
            forecast = self.model.predict(future)
            forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].iloc[-len(test):].to_csv(
                f'csv/{self.name}.csv', index=False)
            self.model.plot(forecast).savefig(f"img/{self.name}-scatter.png")  # 绘制预测效果图
            self.model.plot_components(forecast).savefig(f"img/{self.name}-trend.png")  # 绘制成分趋势图
            self.save_model()
        print(f'score:{self.mse}\nparams:{self.params}')
        return self

    def save_model(self):
        with open(f'model/{self.name}.pkl', 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load_model(name):
        with open(f'model/{name}.pkl', 'rb') as fp:
            return pickle.load(fp)


ph = ProphetTrain(name="PM10")
ph.grid_search()
