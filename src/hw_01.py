#!/usr/bin/env python3

import numpy as np

def _hamming_(mat):
    rows = mat.shape[0]
    _hamming = np.zeros((rows, rows), dtype=int)

    for i in range(1, rows):
        _hamming[i, :i] = np.count_nonzero(mat[i] != mat[:i], axis=1)
        _hamming[:i, i] = _hamming[i, :i]

    return _hamming

data = np.loadtxt('../data/zoo/zoo.data', delimiter=',', dtype='object')
data = data[:, :-1]

_animals = list(data[:, 0])
animals = {_name_: i for i, _name_ in enumerate(_animals)}
vecs = data[:, 1:].astype(int)

hamming = _hamming_(vecs)

while True:
    a0 = input('Nombre animal (o esc para salir): ')
    a1 = input('Nombre animal (o esc para salir): ')

    if a0 == '\x1b' or a1 == '\x1b':
        exit()

    idx0 = animals[a0]
    idx1 = animals[a1]

    print(f'La distancia Hamming entre {a0} y {a1} es: {hamming[idx0, idx1]}')
    print()
    print()
