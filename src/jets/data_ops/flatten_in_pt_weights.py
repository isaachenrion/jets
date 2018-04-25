import numpy as np

def flatten_in_pt_weights(jets):
    pt_min = min(j.pt for j in jets)
    pt_max = max(j.pt for j in jets)

    w = np.zeros(len(jets))
    y = np.array([jet.y for jet in jets])

    bins = 50
    jets_0 = [jet for jet in jets if jet.y == 0]
    pdf, edges = np.histogram([j.pt for j in jets_0], density=True, range=[pt_min - 1, pt_max + 1], bins=bins)
    pts = [j.pt for j in jets_0]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y==0] = inv_w

    jets_1 = [jet for jet in jets if jet.y == 1]
    pdf, edges = np.histogram([j.pt for j in jets_1], density=True, range=[pt_min - 1, pt_max + 1], bins=bins)
    pts = [j.pt for j in jets_1]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y==1] = inv_w

    return w
