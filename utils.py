import numpy as np
import pandas as pd
import pywt


def hour2sincos(df):
    """时间转sin函数"""
    df['time'] = pd.to_datetime(df['time'])
    hour = df['time'].dt.hour
    hour_radians = 2 * np.pi * hour / 24
    df['sin_time'] = np.sin(hour_radians)
    df['cos_time'] = np.cos(hour_radians)
    return df


def wavelet_transform(signal):
    """进行小波变换"""
    # 选用Daubechies 4小波作为基函数
    wavelet = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(signal, wavelet, level=8)
    return coeffs


def threshold(coeffs, threshold_value):
    """对小波系数进行软阈值去噪"""
    thresholded_coeffs = []
    for i in range(len(coeffs)):
        thresholded_coeffs.append(pywt.threshold(coeffs[i], threshold_value))
    return thresholded_coeffs


def reconstruct(coeffs):
    """进行小波重构"""
    reconstructed_signal = pywt.waverec(coeffs, 'db4')
    return reconstructed_signal


def npzeros(y):
    return np.zeros_like(y)


def multiscale_analysis(signal):
    """进行多分辨率分析"""
    # 进行小波变换
    coeffs = wavelet_transform(signal)
    # 对小波系数进行软阈值去噪
    thresholded_coeffs = threshold(coeffs, 0.1)
    # 将小波系数分为8个不同频段
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = thresholded_coeffs
    # 进行小波重构
    reconstructed_signal1 = reconstruct([
        (cA8),
        (cD8),
        (cD7),
        (cD6),
        (cD5),
        (cD4),
        (cD3),
        npzeros(cD2),
        (cD1),
    ])
    # 返回4个不同频段的信号
    return reconstructed_signal1


def normal_distribution(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def sliding_iqr(data, window_size, threshold=1.5):
    """滑动四分位法异常值检测算法

    Parameters:
        data (array_like): 待处理的数据，可以是数组、列表或Series对象
        window_size (int): 滑动窗口大小，用于计算四分位数
        threshold (float): 异常值判别阈值，默认为1.5倍IQR

    Returns:
        mask (array_like): 异常值掩码，True表示为异常值，False表示为正常值
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    rolling_iqr = data.rolling(window_size).quantile(0.75) - data.rolling(window_size).quantile(0.25)
    rolling_median = data.rolling(window_size).median()

    upper_bound = rolling_median + threshold * rolling_iqr
    lower_bound = rolling_median - threshold * rolling_iqr

    mask = (data > upper_bound) | (data < lower_bound)

    return mask


def detect_anomalies(y, window_size, threshold):
    """
    使用滑动窗口方法检测数据中的异常波动

    参数:
    y (numpy.ndarray): 输入的数据序列
    window_size (int): 滑动窗口的大小
    threshold (float): 判定异常波动的阈值

    返回值:
    numpy.ndarray: 一个和输入数据等长的数组，每个位置的值表示对应窗口内的标准差是否超过阈值，超过为 1，否则为 0。
    """

    # 计算每个窗口内的标准差，返回一个和输入数据等长的数组
    # stds = np.array([np.std(y[i:i + window_size]) for i in range(len(y) - window_size + 1)])
    y = pd.Series(y)
    stds = y.rolling(window_size).std()
    # 判断每个窗口内的标准差是否超过阈值，返回一个和输入数据等长的数组
    anomalies = np.where(stds > threshold, True, False)
    anomalies = np.append(anomalies, (np.zeros(len(y) - len(anomalies), dtype=bool)))
    return anomalies