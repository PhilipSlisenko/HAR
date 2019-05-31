import numpy as np
import math


def add_acceleration_vector_column(arr: np.ndarray) -> np.ndarray:
    calc_acc_vect = lambda ser: math.sqrt(np.sum(np.square(ser)))
    acc_vect = np.apply_along_axis(calc_acc_vect, 1, arr)
    return np.insert(arr, arr.shape[1], acc_vect, axis=1)


def normalize_and_drop_g(arr: np.ndarray) -> np.ndarray:
    """
    x, y, z axis accelerations divided by total acceleration g
    afterwards g dropped
    """
    norm_arr = np.empty((arr.shape[0], arr.shape[1] - 1), dtype=float)
    norm_arr[:, 0] = arr[:, 0] / arr[:, 3]
    norm_arr[:, 1] = arr[:, 1] / arr[:, 3]
    norm_arr[:, 2] = arr[:, 2] / arr[:, 3]
    return norm_arr


def rms(arr: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(arr)))


def calculate_new_features(arr: np.ndarray) -> np.ndarray:
    """
    Receives: np.ndarray (columns: normalized acceleration along x/g, y/g, z/g).
    Returns: processed nd.array which contains following columns: rms, mean, std of each axis
    """
    new_arr = np.apply_along_axis(rms, 0, arr)
    new_arr = np.append(new_arr, np.mean(arr, axis=0))
    new_arr = np.append(new_arr, np.std(arr, axis=0))
    return new_arr


def generate_new_features(arr: np.ndarray) -> np.ndarray:
    processed_arr = add_acceleration_vector_column(arr)
    processed_arr = normalize_and_drop_g(processed_arr)
    processed_arr = calculate_new_features(processed_arr)
    return processed_arr