"""
A collection of functions that transform between various representations of
quantum channels

@author: ebutler414
"""

import numpy as np
from numpy import ndarray
from scipy.linalg import eig
from typing import Union, Tuple
from warnings import warn

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
    dimension_squared = rho.size
    dimension = np.sqrt(dimension_squared)
    assert dimension.is_integer(), \
    "dimension must be a square number since is d_sys**2"
    # safely convert float from sqrt to int so can be used as a shape
    dimension = int(dimension)
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

def transform_choi_to_kraus(choi_matrix: ndarray,
        return_left_operators: bool = False
        ) -> Union[ndarray[ndarray],Tuple[ndarray[ndarray]]]:
    """
    Transforms a choi matrix expressed as a rank 4 tensor into an array
    orthogonal Kraus operators, [[kraus_0],[kraus_1]...[kraus_d**2]]
    """

    assert choi_matrix.ndim == 4, "must be rank 4 tensor"
    assert choi_matrix.shape[0] == choi_matrix.shape[1] == \
        choi_matrix.shape[2] == choi_matrix.shape[3], \
        "all of the indices must be the same dimension"

    dimension = choi_matrix.shape[0]

    choi_vectorised = vectorise_superoperator(choi_matrix)
    eigenvalues,eigenvectors = eig(choi_vectorised)

    if not np.all(eigenvalues >= 0):
        warn('some eigenvalues are negative, this is not a CP map')

    # stealing wood notation
    lambda_values = np.sqrt(eigenvalues)

    kraus_matrices = np.zeros((dimension**2,dimension,dimension),
        dtype='complex128')
    if return_left_operators:
        kraus_matrices_left = np.zeros((dimension**2,dimension,dimension),
            dtype='complex128')

    # someone who is good at numpy indexing would be able to remove this
    # for-loop. Unfortunately I cannot as I, as a matter of fact, am not good at
    # anything. (ah in fairness figuring that out would be a pain in the hole)
    for i in range(lambda_values.size):
        normalised_eigenvector = eigenvectors[:,i]
        eigenvector = lambda_values[i] * normalised_eigenvector
        kraus_matrix = eigenvector.reshape(dimension,dimension)
        if return_left_operators:
            kraus_matrices_left[i,:,:] = kraus_matrix.conjugate().T
        kraus_matrices[i,:,:] = kraus_matrix
    if return_left_operators:
        return kraus_matrices,kraus_matrices_left
    return kraus_matrices

def evolve_using_kraus_operators(
                                kraus_operators: ndarray,
                                density_matrix: ndarray,) -> ndarray:
    """
    Takes an array of Kraus operators, and performes the evolution of the
    density matrix. The Kraus operators need to be given as an array where the
    first index is the index of the kraus operators, and the second and third
    index represent the first and second index of the Kraus operators
    respectively.
    """

    assert density_matrix.ndim == 2, "density matrix must be a matrix"
    assert density_matrix.shape[0] == density_matrix.shape[1], \
        "density matrix must be square"
    dimension = density_matrix.shape[0]

    assert kraus_operators.shape[1] == kraus_operators.shape[2] \
        == dimension, "Kraus operators should be same shape as density matrix"
    assert kraus_operators.shape[0] <= dimension ** 2, \
        ("For a system of dimension {} there should be no more than {} Kraus"
        "operators".format(dimension,dimension**2))

    final_dm = np.zeros((kraus_operators.shape[1],kraus_operators.shape[2]),
        dtype='complex128')
    for i in range(kraus_operators.shape[0]):
        kraus_operator = kraus_operators[i]
        kraus_dagger = kraus_operator.conjugate().T
        intermediate = np.matmul(kraus_operator,density_matrix)
        summand = np.matmul(intermediate,kraus_dagger)
        final_dm += summand
    return final_dm
