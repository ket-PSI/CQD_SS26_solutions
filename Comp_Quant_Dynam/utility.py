import numpy as np   # standard numerics library
import math

def example_func(x):
    """
    Example function to demonstrate the repository structure.
    Returns the ground state wavefunction of the quantum harmonic oscillator at position 'x' in numerical units.
    """

    return 1 / np.pi ** (1 / 4) * np.exp(-x ** 2 / 2)

#################### Solution sheet 2 ####################


def create_xvals(L, npoints, endpoint=True):
    """
    Creates a grid of 'npoints' evenly spaced values between -L/2 and L/2.
    The 'endpoint' parameter determines whether the endpoint L/2 is included in the grid.
    Returns the grid of x values and the grid spacing dx.
    """
    xvals = np.linspace(-L / 2, L / 2, npoints, endpoint=endpoint)
    dx = xvals[1] - xvals[0]
    return xvals, dx

#################### Solution sheet 3 ####################


def FT(psi, x, k):
    """
    Computes the discrete Fourier transform of the wavefunction `psi` defined on the grid `x` to the momentum space grid `k`.
    """
    npoints = len(x)
    assert len(k) == npoints, "Length of k must match length of x"
    return np.sum(psi * np.exp(-1j * np.outer(k, x)), axis=1)

def iFT(phi, x, k):
    """
    Computes the inverse discrete Fourier transform of the momentum space wavefunction `phi` defined on the grid `k` to the position space grid `x`.
    """
    npoints = len(x)
    assert len(k) == npoints, "Length of k must match length of x"
    return 1 / npoints * np.sum(phi * np.exp(1j * np.outer(x, k)), axis=1)

def gaussian_wave_packet(x, x0, sigma, p0):
    """
    Creates a Gaussian wave packet centered at position `x0` with width `sigma` and momentum `p0`.
    """
    norm = 1 / np.sqrt(np.sqrt(2 * np.pi) * sigma)
    # p0 * x0 is a global phase factor that we can ignore, so we can omit it
    #  in the expression for the wave packet.
    return norm * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2) + 1j * p0 * x)

def create_tvecs(tsteps, dt):
    """
    Creates a time vector for steps'' time steps with time step size 'dt'.
    Returns the time vector of length tsteps+1, starting from 0 to tsteps*dt.
    """
    return np.linspace(0, tsteps * dt, tsteps + 1) # will have length tsteps + 1


##################### Exercise sheet 4 ###################

def idx2state(N1, N2, i):
    """
    Converts a single index `i` to a 'state' in the product Hilbert space |n1, n2> of dimension N1 x N2.
    """
    assert i >= 0 and i < N1 * N2, "Index out of bounds"
    n1 = i // N2
    n2 = i % N2
    state = [n1, n2]
    return state 
 
def state2idx(N1, N2, state):
    """
    Converts a `state` |n1, n2> from the product Hilbert space of dimension `N1 x N2` to a single index `i`.
    """
    n1 = state[0]
    n2 = state[1]
    if n1 < 0 or n1 >= N1 or n2 < 0 or n2 >= N2:
        i = -1 # return -1 if the state is out of bounds
    else:
        i = n1 * N2 + n2
    return i

##################### Solution sheet 4 ###################

def create_coherent_state(N, alpha):
    """
    Creates a coherent state |alpha> in the Fock basis of dimension `N` with complex amplitude `alpha`.
    The coherent state is defined as:
    |alpha> = exp(-|alpha|^2/2) sum_{n=0}^{N-1} (alpha^n / sqrt(n!)) |n>
    """

    nvec = np.arange(N)
    state = np.exp(-np.abs(alpha) ** 2 / 2) * np.power(alpha, nvec) / np.sqrt(factorial(nvec))
    return state

def expectation_value(state, operator):
    """
    Computes the expectation value of an `operator` in a given `state`.
    """

    return np.vdot(state, operator @ state)