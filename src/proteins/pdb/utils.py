import string
import numpy as np

def relevant_atom(residue):
    '''Get the atom from a residue that is relevant for measuring residue-residue distances.
    In all amino acids this is CB, except for glycine where it is CA. If neither of these
    is present, return None'''
    if residue.get_resname().upper() == 'GLY':
        atom_id = 'CA'
    else:
        atom_id = 'CB'
    try:
        atom = residue[atom_id]
    except KeyError:
        atom = None
    return atom

def relevant_chain(structure, chain_id):
    '''Given a structure and a chain id, return that chain from the structure.
    This is robust to upper and lower case.'''
    model = structure[0]
    try:
        chain = model[chain_id.upper()]
    except KeyError:
        try:
            chain = model[chain_id.lower()]
        except KeyError:
            raise ValueError("Structure {} has no chain {}. It only has {}".format(structure.get_id(), chain_id, list(structure.get_chains())))
    return chain

def get_seq_and_coords(structure, chain_id, alphabet, distances=False):
    '''Given a structure and chain id, return the sequence as a one-hot numpy
    array, and the coordinates of the residues as a numpy array.
    Optionally, return the matrix of distances between the residues.'''
    chain = relevant_chain(structure, chain_id)

    residues = list(chain.get_residues())
    residues = [r for r in residues if r.get_resname() in alphabet]

    # sequence
    sequence = [r.get_resname() for r in residues]
    seq_vec = _string_vectorizer(sequence, alphabet)
    if not np.all(seq_vec.sum(1) == 1):
        bad_indices = np.where(seq_vec.sum(1) != 1)[0]
        bad_residues = [r.get_resname() for i, r in enumerate(residues) if i in bad_indices]
        out_str = "Sequence {}/{} has not been correctly one-hot encoded.\n".format(pdb_id, chain_id)
        out_str += "Indices {} in the sequence were incorrectly encoded\n".format(', '.join(bad_indices))
        our_str += "These correspond to the following residues: {}".format(', '.join(bad_residues))
        raise ValueError(out_str)

    # coords
    coords = np.zeros((len(residues), 3))
    for i, r1 in enumerate(residues):
        a1 = relevant_atom(r1)
        if a1 is not None:
            xyz = a1.get_coord()
            coords[i] = xyz
        else:
            coords[i] = [np.inf, np.inf, np.inf]

    # distances
    if distances:
        dists = np.zeros((len(residues), len(residues)))
        for i, r1 in enumerate(residues):
            a1 = relevant_atom(r1)
            for j, r2 in enumerate(residues):
                a2 = relevant_atom(r2)
                if a1 is not None and a2 is not None:
                    dists[i, j] = a1 - a2
                else:
                    dists[i, j] = np.inf
        return seq_vec, coords, dists
    return seq_vec, coords

def _string_vectorizer(strng, alphabet=string.ascii_uppercase):
    '''Given a string and alphabet, return a one-hot representation as a numpy array'''
    vector = np.array([[0 if char != letter else 1 for char in alphabet]
                  for letter in strng])
    return vector
