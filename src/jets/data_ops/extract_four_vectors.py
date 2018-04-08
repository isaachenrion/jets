import numpy as np

def extract_four_vectors(four_vectors):
    ''' Convert an array of four-vectors into 7-dim jet constituent representation.
    '''
    assert four_vectors.shape[1] == 4

    content = np.zeros((len(four_vectors), 7))

    for i in range(len(four_vectors)):
        px = four_vectors[i, 0]
        py = four_vectors[i, 1]
        pz = four_vectors[i, 2]
        E = four_vectors[i, 3]
        total_E = sum(four_vectors[:, 3])

        p = (four_vectors[i, 0:3] ** 2).sum() ** 0.5
        eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
        theta = 2 * np.arctan(np.exp(-eta))
        pt = p / np.cosh(eta)
        phi = np.arctan2(py, px)

        content[i, 0] = p
        content[i, 1] = eta if np.isfinite(eta) else 0.0
        content[i, 2] = phi
        content[i, 3] = E
        content[i, 4] = E / total_E
        content[i, 5] = pt if np.isfinite(pt) else 0.0
        content[i, 6] = theta if np.isfinite(theta) else 0.0
        #content[i, 7] = px
        #content[i, 8] = py
        #content[i, 9] = pz

    return content
