from typing import List
import numpy as np
from src.generate_new_features import generate_new_features


def read_file(fname: str) -> np.ndarray:
    arr = np.genfromtxt(fname, dtype=float, delimiter=',', skip_header=2)
    return arr


def divide_into_windows(arr: np.ndarray, window_size: int) -> List[np.ndarray]:
    num_windows = int(arr.shape[0] / window_size)
    windows = list()
    for i in range(num_windows):
        windows.append(arr[i * window_size:i * window_size + window_size])
    return windows


def transform_windows_to_samples_with_labels(windows: List[np.ndarray], label: int) -> np.ndarray:
    samples = list()
    for window in windows:
        samples.append(generate_new_features(window))
    samples = np.array(samples)
    label_column = np.full((samples.shape[0], 1), float(label))
    samples = np.append(samples, label_column, axis=1)
    return samples


def raw_to_intermediate_with_label(fname: str, label: int, window_size: int = 6):
    arr = read_file(fname)
    windows = divide_into_windows(arr, window_size)
    arr = transform_windows_to_samples_with_labels(windows, label)
    return arr