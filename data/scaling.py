import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale(min_val, max_val, data: np.ndarray):
    data = data.copy()
    for i in range(data.size):
        data[i] = (data[i]-min_val)/(max_val-min_val)
    return data


def normalize_arround(data: np.ndarray, window_size: int = 700) -> np.ndarray:
    offset = window_size//2
    new_data = data.copy()

    for i in range(data.size):
        left = max(i - offset, 0)
        right = min(i+(window_size-offset), data.size)
        min_val = data[left:right].min()
        max_val = data[left:right].max()
        new_data[i] = (data[i]-min_val)/(max_val-min_val)

    return new_data, min_val, max_val


def normalize_behind(data: np.ndarray, window_size: int = 500) -> np.ndarray:
    new_data = data.copy()

    for i in range(data.size):
        left = max(i - window_size, 0)
        right = left + window_size
        min_val = data[left:right+1].min()
        max_val = data[left:right+1].max()
        new_data[i] = (data[i]-min_val)/(max_val-min_val)

    return new_data, min_val, max_val


def normalize_until_current_step(data: np.ndarray) -> np.ndarray:
    new_data = data.copy()

    for i in range(len(data)):
        min_val = data[0:i+1].min()
        max_val = data[0:i+1].max()
        new_data[i] = (data[i]-min_val)/(max_val-min_val)

    return new_data, min_val, max_val


def normalize(data: np.ndarray, range=(0, 1)) -> np.ndarray:
    scaler = MinMaxScaler(range)
    norm_data = data.reshape(-1, 1)

    # Train the Scaler with training data and smooth data
    scaler.fit(norm_data)
    norm_data = scaler.transform(norm_data).reshape(-1)

    return norm_data, scaler


def smooth_data_curves(train_data: np.ndarray) -> np.ndarray:
    EMA = 0.0
    gamma = 0.8
    for ti in range(len(train_data)):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA
    return train_data
