"""
A collection of functions that transform between various representations of
quantum channels

@author: ebutler414
"""

import numpy as np
from numpy import ndarray

def density_matrix_to_liouville(rho: ndarray) -> ndarray:
    """
    Transforms a density matrix in hilbert space from hilbert space to
    vectorised liouville space. i.e. takes a (d,d) -> (d**2)
    """
    assert rho.ndim == 2, "density matrix must be a matrix"
    assert rho.shape[0] == rho.shape[1], "density matrix must be square"
    dimension = rho.shape[0]
    return rho.reshape(dimension**2)

def density_matrix_to_hilbert(rho: ndarray) -> ndarray:
    """
    Transforms a vectorised density matrix in liouville space from liouville
    space to hilbert space. i.e. takes a (d**2) vector and returns (d,d) matrix
    """
    assert rho.ndim == 1, "state vector must be a vector"
    dimension = rho.size
    return rho.reshape(dimension,dimension)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this section lifted from oqupy.operators, with eoin's additional notes 
# in docstrings

def commutator(operator: ndarray) -> ndarray:
    """Construct commutator superoperator from operator. For use generating the
    liouvillian. The resultant superoperator is vectorised into the form of
    (d**2,d**2)"""
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim)) \
            - np.kron(np.identity(dim), operator.T)

def acommutator(operator: ndarray) -> ndarray:
    """Construct anti-commutator superoperator from operator. The resultant
    superoperator is vectorised into the form of (d**2,d**2)"""
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim)) \
            + np.kron(np.identity(dim), operator.T)

def left_super(operator: ndarray) -> ndarray:
    """Construct the *right acting* superoperator which sits to the left of the
    unvectorised density matrix. The resultant superoperator is vectorised into
    the form of (d**2,d**2) """
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim))

def right_super(operator: ndarray) -> ndarray:
    """Construct the *left acting* superoperator which sits to the right of the
    unvectorised density matrix. The resultant superoperator is vectorised into
    the form of (d**2,d**2) """
    dim = operator.shape[0]
    return np.kron(np.identity(dim), operator.T)

def left_right_super(
        left_operator: ndarray,
        right_operator: ndarray) -> ndarray:
    """Construct left and right acting superoperator from operators. If the left
    operator and right operator are U and U^{dagger} respectively, this performs
    a unitary rotation of the density matrix. The resultant superoperator is
    vectorised into the form of (d**2,d**2) """
    return np.kron(left_operator, right_operator.T)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def vectorise_superoperator(A: ndarray) -> ndarray:
    """
    takes a rank 4 superoperator that is unvectorised (d,d,d,d), and vectorises
    it, -> (d**2,d**2)
    """

    assert A.ndim == 4, "must be rank 4 tensor"
    assert A.shape[0] == A.shape[1] == A.shape[2] == A.shape[3], \
        "all of the indices must be the same dimension"
    dimension = A.shape[0]
    return A.reshape(dimension**2,dimension**2)

def unvectorise_superoperator(A: ndarray) -> ndarray:
    """
    takes a rank 2 superoperator that is vectorised (d**2,d**2), and
    unvectorises it, (d,d,d,d)
    """

    assert A.ndim == 2, "must be rank 2 tensor"
    assert A.shape[0] == A.shape[1], "matrix must be square"
    dimension_squared = A.shape[0]
    dimension = np.sqrt(dimension_squared)
    assert dimension.is_integer(), \
    "dimension must be a square number since is d_sys**2"
    dimension = int(dimension)
    return A.reshape(dimension,dimension,dimension,dimension)

def transform_superoperator_to_choi(superoperator: ndarray) -> ndarray:
    """
    Transformes a superoperator expressed as a rank 4 tensor into a choi matrix
    expressed as a rank four tensor.

    The transformation is, S^{ijkl} -> S^{kilj} = Λ^{ijkl}
    """
    assert superoperator.ndim == 4, "must be rank 4 tensor"
    assert superoperator.shape[0] == superoperator.shape[1] == \
        superoperator.shape[2] == superoperator.shape[3], \
        "all of the indices must be the same dimension"

    choi_matrix = np.swapaxes(superoperator,0,2)
    choi_matrix = np.swapaxes(choi_matrix,2,3)
    choi_matrix = np.swapaxes(choi_matrix,1,3)
    return choi_matrix


