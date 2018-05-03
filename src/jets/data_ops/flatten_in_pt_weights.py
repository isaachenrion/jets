import numpy as np

def get_weights_for_flatness_in_pt(pts, pt_min, pt_max, bins):
    pdf, edges = np.histogram(pts, density=True, range=[pt_min, pt_max], bins=bins)
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    return inv_w

def flatten_in_pt_weights(jets, pt_min=None, pt_max=None):
    bins = 50
    if pt_min is None:
        pt_min = min(j.pt for j in jets) - 1
    if pt_max is None:
        pt_max = max(j.pt for j in jets) + 1

    w = np.zeros(len(jets))
    y = np.array([jet.y for jet in jets])

    pts_0 = [jet.pt for jet in jets if jet.y == 0]
    pts_1 = [jet.pt for jet in jets if jet.y == 1]

    w_0 = get_weights_for_flatness_in_pt(pts_0, pt_min, pt_max, bins)
    w_1 = get_weights_for_flatness_in_pt(pts_1, pt_min, pt_max, bins)

    w[y==0] = w_0
    w[y==1] = w_1

    return w

''' THIS IS WRONG BUT WHAT I USED TO GET THE GOOD RESULTS
w_0 = get_weights_for_flatness_in_pt(pts_0, pt_min, pt_max, bins)
w_1 = get_weights_for_flatness_in_pt(pts_1, pt_min, pt_max, bins)

for i, (iw, jet) in enumerate(zip(w_0, jets)):
    if jet.y == 0:
        w[i] = iw
for i, (iw, jet) in enumerate(zip(w_1, jets)):
    if jet.y == 1:
        w[i] = iw
'''
