import string
import numpy as np

def relevant_atom(residue):
    if residue.get_resname() == 'GLY':
        atom = 'CA'
    else:
        atom = 'CB'
    return atom

def get_distances(structure):

    residues = list(structure.get_residues())
    sequence = list(map(lambda r: r.get_resname(), residues))
    dists = np.zeros((len(residues), len(residues)))
    for i, r1 in enumerate(residues):
        a1 = relevant_atom(r1)
        for j, r2 in enumerate(residues):
            a2 = relevant_atom(r2)
            dists[i, j] = r1[a1] - r2[a2]

    return sequence, dists

def _string_vectorizer(strng, alphabet=string.ascii_uppercase):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return vector
