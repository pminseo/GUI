# https://gist.github.com/KeremTurgutlu/68feb119c9dd148285be2e247267a203
import numpy as np

def nn_interpolate(A, new_size):
    """Vectorized Nearest Neighbor Interpolation"""
    old_size = A.shape
    row_ratio, col_ratio = np.array(new_size)/np.array(old_size)
    
    # row wise interpolation 
    row_idx = (np.ceil(range(0, new_size[0])/row_ratio - 0.9999)).astype(int)

    # column wise interpolation
    col_idx = (np.ceil(range(0, new_size[1])/col_ratio - 0.9999)).astype(int)

    final_matrix = A[row_idx, :][:, col_idx]
    return final_matrix.copy()
