import numpy as np   # standard numerics library


#################### Solution sheet 2 ####################


def t_evol_eigenbasis(init_coeffs, t, evals, evecs):
    """
    Returns the state in the computational basis at time 't' given the initial state coefficients 'init_coeffs'
    in the eigenbasis for a Hamiltonian with eigenvalues 'evals' and eigenvectors 'evecs'.
    """

    phase_factors = np.exp(-1j * evals * t) # compute the phase factors for each eigenstate
    evolved_eigen_coeffs = init_coeffs * phase_factors
    return evecs @ evolved_eigen_coeffs

def init_coeffs_eigenbasis(psi0, evecs):
    """
    Returns the coefficients of the initial state 'psi0' in the eigenbasis given by 'evecs'.
    """

    return np.conjugate(evecs.T) @ psi0