import logging
import numpy as np

def crop(X, y, return_cropped_indices=False, pileup=False):
    logging.warning("Cropping...")
    if pileup:
        pt_min, pt_max, m_min, m_max = 300, 365, 150, 220
    else:
        pt_min, pt_max, m_min, m_max = 250, 300, 50, 110
    indices = [i for i, j in enumerate(X) if pt_min < j["pt"] < pt_max and m_min < j["mass"] < m_max]
    cropped_indices = [i for i, j in enumerate(X) if i not in indices]
    logging.warning("{} (selected) + {} (cropped) = {}".format(len(indices), len(cropped_indices), (len(indices) + len(cropped_indices))))
    X_ = [j for j in X if pt_min < j["pt"] < pt_max and m_min < j["mass"] < m_max]
    y_ = [y[i] for i, j in enumerate(X) if pt_min < j["pt"] < pt_max and m_min < j["mass"] < m_max]

    y_ = np.array(y_)

    # Weights for flatness in pt
    w = np.zeros(len(y_))

    X0 = [X_[i] for i in range(len(y_)) if y_[i] == 0]
    pdf, edges = np.histogram([j["pt"] for j in X0], density=True, range=[pt_min, pt_max], bins=50)
    pts = [j["pt"] for j in X0]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y_==0] = inv_w

    X1 = [X_[i] for i in range(len(y_)) if y_[i] == 1]
    pdf, edges = np.histogram([j["pt"] for j in X1], density=True, range=[pt_min, pt_max], bins=50)
    pts = [j["pt"] for j in X1]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y_==1] = inv_w

    if return_cropped_indices:
        return X_, y_, cropped_indices, w
    return X_, y_, w
