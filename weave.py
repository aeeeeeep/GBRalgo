import pickle
from abc import abstractmethod
import pywt
import pandas as pd
from utils import *
import numpy as np
# from fbprophet import Prophet
# from fbprophet.diagnostics import cross_validation, performance_metrics
from matplotlib import pyplot as plt
from warnings import simplefilter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

micro1 = pd.read_excel('./resource/micro1.xlsx')
micro2 = pd.read_excel('./resource/micro2.xlsx')
standard = pd.read_excel('./resource/standard.xlsx')
std_col = ['PM₂.₅ μg/m³','PM₁₀ μg/m³','NO₂ μg/m³','temperature','humidity']
micro1.iloc[:, 2:] = micro1.iloc[:, 2:].clip(lower=0)
micro2.iloc[:, 2:] = micro2.iloc[:, 2:].clip(lower=0)
standard.iloc[:, 2:] = standard.iloc[:, 2:].clip(lower=0)
# for i in range(5):
    # mm = MinMaxScaler()
    # y1 = multiscale_analysis(standard[std_col[i]]-micro2[micro2.columns[i + 2]])
    # y1 = (y-y.min())/(y.max()-y.min())
    # plt.plot(standard[std_col[i]]-micro2[micro2.columns[i + 2]], linestyle='--', alpha=0.3)
    # plt.plot(y1, label='y1')
    # anomalies = detect_anomalies(y1, window_size=96, threshold=2)
    # plt.scatter(np.where(anomalies)[0], y1[anomalies])
    # plt.axhline(0,color='k',linestyle='--')
    # plt.title(micro2.columns[i + 2])
    # plt.legend()
    # plt.show()

y = multiscale_analysis(standard['PM₂.₅ μg/m³'])
plt.figure(figsize=(10,3),dpi=500)
plt.plot(standard['PM₂.₅ μg/m³'], linewidth=0.5, linestyle='--', alpha=0.3,label='raw')
plt.plot(y,label='y',linewidth=0.5)
plt.legend()
plt.show()