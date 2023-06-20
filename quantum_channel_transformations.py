'''
A collection of functions that transform between various representations of
quantum channels

@author: ebutler414
'''

import numpy as np

def density_matrix_to_liouville(rho: np.ndarray) -> np.ndarray:
    '''
    Transforms a density matrix in hilbert space from hilbert space to
    vectorised liouville space. i.e. takes a (d,d) -> (d**2)
    '''
    assert rho.ndim == 2, 'density matrix must be a matrix'
    assert rho.shape[0] == rho.shape[1], 'density matrix must be square'
    dimension = rho.shape[0]
    return rho.reshape(dimension**2)

def density_matrix_to_hilbert(rho: np.ndarray) -> np.ndarray:
    '''
    Transforms a vectorised density matrix in liouville space from liouville
    space to hilbert space. i.e. takes a (d**2) vector and returns (d,d) matrix
    '''
    assert rho.ndim == 1, 'state vector must be a vector'
    dimension = rho.size
    return rho.reshape(dimension,dimension)

def construct_superoperator(A: np.ndarray) -> np.ndarray:
    '''
    takes a linear operator A and transforms it to a superoperator in vectorised
    liouville space, i.e. a (d**2,d**2) matrix. This is just lifted from tempo

    this introduction is wrong, please fix
    '''
    # atm this is the commutator
    dim = A.shape[0]
    return np.kron(A, np.identity(dim)) - np.kron(np.identity(dim), A.T)

def vectorise_superoperator(A: np.ndarray) -> np.ndarray:
    '''
    takes a rank 4 superoperator that is unvectorised (d,d,d,d), and vectorises
    it, -> (d**2,d**2)
    '''

    assert A.ndim == 4, 'must be rank 4 tensor'
    assert A.shape[0] == A.shape[1] == A.shape[2] == A.shape[3], \
        'all of the indices must be the same dimension'
    dimension = A.shape[0]
    return A.reshape(dimension**2,dimension**2)

def unvectorise_superoperator(A: np.ndarray) -> np.ndarray:
    '''
    takes a rank 2 superoperator that is vectorised (d**2,d**2), and
    unvectorises it, (d,d,d,d)
    '''

    assert A.ndim == 2, 'must be rank 2 tensor'
    assert A.shape[0] == A.shape[1], 'matrix must be square'
    dimension_squared = A.shape[0]
    dimension = np.sqrt(dimension_squared)
    assert dimension.is_integer(), \
    'dimension must be a square number since is d_sys**2'
    return A.reshape(dimension,dimension,dimension,dimension)

def transform_superoperator_to_choi(superoperator: np.ndarray) -> np.ndarray:
    '''
    Transformes a superoperator expressed as a rank 4 tensor into a choi matrix
    expressed as a rank four tensor.

    The transformation is, S^{ijkl} -> S^{kilj} = Λ^{ijkl}
    '''
    assert superoperator.ndim == 4, 'must be rank 4 tensor'
    assert superoperator.shape[0] == superoperator.shape[1] == \
        superoperator.shape[2] == superoperator.shape[3], \
        'all of the indices must be the same dimension'

    choi_matrix = np.swapaxes(superoperator,0,2)
    choi_matrix = np.swapaxes(choi_matrix,2,3)
    choi_matrix = np.swapaxes(choi_matrix,1,3)
    return choi_matrix
