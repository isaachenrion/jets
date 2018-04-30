import numpy as np
def unwrap(x):
    return np.array(x.detach().cpu().numpy())
