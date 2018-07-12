import string
import numpy as np

def unknown_mask(contact_matrix):
    x, y = np.where(contact_matrix == -1)
    mask = np.ones_like(contact_matrix)
    mask[x,y] = 0
    return mask

def _string_vectorizer(strng, alphabet=string.ascii_uppercase):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return vector

def process_sequence_string(seq):
    return np.array(_string_vectorizer(seq))[:, :20]

class Protein:
    def __init__(
            self,
            sequence=None,
            contact_matrix=None,
            acc=None,
            ss3=None
            ):
        self.string_sequence = sequence
        self.sequence=process_sequence_string(sequence)
        self.contact_matrix=contact_matrix.astype('float32')
        self.acc=acc
        self.ss3=ss3

    @property
    def mask(self):
        try:
            return self._mask
        except AttributeError:
            self._mask = unknown_mask(self.contact_matrix)
            return self._mask

    def __len__(self):
        return len(self.sequence)

    @classmethod
    def from_record(cls, record):
        return cls(sequence=record['sequence'], contact_matrix=record['contactMatrix'], acc=record['ACC'], ss3=record['SS3'])
