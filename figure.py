import pickle
from abc import abstractmethod
import pandas as pd
import numpy as np
# from fbprophet import Prophet
# from fbprophet.diagnostics import cross_validation, performance_metrics
from matplotlib import pyplot as plt
from utils import *
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
#     plt.figure(figsize=(10, 3), dpi=300)
#     mm = MinMaxScaler()
#     plt.plot(micro1[micro1.columns[i+2]],label='1_'+micro1.columns[i+2],alpha=0.5,linewidth=0.5)
#     plt.plot(standard[std_col[i]],label='s_'+std_col[i],linewidth=0.5)
#     # plt.plot(standard['temperature'],label='temp',alpha=0.5)
#     # plt.scatter(standard['temperature'][1:], micro1['PM2.5'],label='micro1',alpha=0.5)
#     plt.legend()
#     plt.show()
window_size = [48,48,72,24,24]
threshold = [2.5,3,5,8,5]
for i in [0,1]:
    plt.figure(figsize=(10, 3), dpi=300)
    # mm = MinMaxScaler()
    # mask = sliding_iqr(((standard[std_col[i]]-micro2[micro2.columns[i+2]])), window_size=window_size[i], threshold=threshold[i])
    mask = np.array(micro2['cls_'+micro2.columns[i+2]]-1,dtype=bool)
    plt.scatter(np.where(mask)[0], ((standard[std_col[i]]-micro2[micro2.columns[i+2]]))[mask], c='r')
    # print(np.where(mask)[0],' ',micro2.columns[i+2])
    plt.plot(((standard[std_col[i]]-micro2[micro2.columns[i+2]])),label=micro2.columns[i+2],linewidth=0.5)
    # micro2[micro2.columns[i + 2]].loc[mask,] = standard[std_col[i]].loc[mask]
    # plt.plot(((standard[std_col[i]]-micro2[micro2.columns[i+2]])),label='change_'+micro2.columns[i+2],linewidth=0.5)
    plt.axhline(0,color='k',linestyle='--')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 3), dpi=300)
    plt.plot(standard[std_col[i]],label=std_col[i],linewidth=0.5)
    plt.plot(micro2[micro2.columns[i+2]],label=micro2.columns[i+2],linewidth=0.5)
    plt.legend()
    plt.show()