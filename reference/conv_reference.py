import numpy as np

def conv(x, w):
    return np.array([np.sum(x[i:i+3,j:j+3]*w, dtype=np.float32)
                     for i in range(26) for j in range(26)], dtype=np.float32)
