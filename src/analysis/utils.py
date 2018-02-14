import numpy as np

def exponential_moving_average(x, alpha):
    s = np.zeros_like(x)
    s[0] = x[0]
    for t in range(len(x) - 1):
        s[t+1] = alpha * x[t+1] + (1 - alpha) * s[t]
    return s
