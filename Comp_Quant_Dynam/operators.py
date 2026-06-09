import numpy as np
from scipy import sparse


##################### Solution sheet 4 ####################


def diagonal_op_sparse(arr, offsets = 0):
    """
    Returns a sparse diagonal matrix from a given 1D array `arr`.
    The diagonal matrix has the elements of `arr` on its diagonal and zeros elsewhere.
    """
    
    return sparse.diags_array(arr, offsets=offsets)

def n_party_op_sparse(local_dims, idx, op):
    """
    Returns a sparse operator for an n-party system, where `local_dims` is a list of the local dimensions of each party, `idx` is the index of the party on which the local operator `op` acts,
    and `op` is the local operator represented as a 2D array.
    """
    
    assert idx < len(local_dims), "Index of the local operator must be less than the total number of parties."
    assert op.shape == (local_dims[idx], local_dims[idx]), "Local operator must have the correct shape corresponding to the local dimension of the party."

    eye_left = sparse.eye_array(np.prod(local_dims[:idx])) # identity operator on the left of the local operator
    eye_right = sparse.eye_array(np.prod(local_dims[idx+1:])) # identity operator on the right of the local operator
    full_op = sparse.kron(eye_left, sparse.kron(op, eye_right)) # full operator is the Kronecker product of the left identity, local operator, and right identity
    return full_op

def a_operator_sparse(N):
    """
    Returns the annihilation ladder operator for a system with a local dimension `N` as a sparse matrix.
    The annihilation operator is defined as:
    a |n> = sqrt(n) |n-1> for n > 0
    a |0> = 0
    """
    vals = np.sqrt(np.arange(1, N)) # non-zero elements of the annihilation operator
    a_op = diagonal_op_sparse(vals, offsets=1) # create a sparse diagonal matrix with the non-zero elements on the first upper diagonal
    return a_op

def adag_operator_sparse(N):
    """
    Returns the creation ladder operator for a system with a local dimension `N` as a sparse matrix.
    The creation operator is defined as:
    a^dag |n> = sqrt(n+1) |n+1> for n < N-1
    a^dag |N-1> = 0
    """
    a_op = a_operator_sparse(N) # get the annihilation operator
    adag_op = a_op.transpose().conjugate() # the creation operator is the Hermitian conjugate of the annihilation operator
    return adag_op

def n_operator_sparse(N):
    """
    Returns the number operator for a system with a local dimension `N` as a sparse matrix.
    The number operator is defined as:
    n |n> = n |n>
    """
    n_vals = np.arange(N, dtype=float) # eigenvalues of the number operator
    n_op = diagonal_op_sparse(n_vals) # create a sparse diagonal matrix with the eigenvalues on the diagonal
    return n_op

def x_operator_sparse(N):
    """
    Returns the position operator for a system with a local dimension `N` as a sparse matrix.
    The position operator is defined as:
    x |n> = sqrt(1/2) (a + a^dag) |n>
    """

    a_op = a_operator_sparse(N) # get the annihilation operator
    adag_op = adag_operator_sparse(N) # get the creation operator
    x_op = np.sqrt(1/2) * (a_op + adag_op) # the position operator is the sum of the annihilation and creation operators
    return x_op

def p_operator_sparse(N):
    """
    Returns the momentum operator for a system with a local dimension `N` as a sparse matrix.
    The momentum operator is defined as:
    p |n> = i * sqrt(1/2) (a^dag - a) |n>
    """

    a_op = a_operator_sparse(N) # get the annihilation operator
    adag_op = adag_operator_sparse(N) # get the creation operator
    p_op = 1j * np.sqrt(1/2) * (adag_op - a_op) # the momentum operator is the difference of the creation and annihilation operators, multiplied by i and sqrt(1/2)
    return p_op

def n_proj_operator_sparse(local_dims, idx, n):
    """
    Returns the projector operator for the `n`-th state of the `idx`-th party in an system with local dimensions given by `local_dims` as a sparse matrix.
    The projector operator is defined as:
    P_n = |n><n| for the n-th state of the idx-th party, and acts as the identity on the other parties.
    """

    proj_arr = np.eye(1, local_dims[idx], n)
    proj_op = diagonal_op_sparse(proj_arr.flatten()) # create a sparse diagonal matrix with the non-zero element on the diagonal corresponding to the n-th state
    projector_full = n_party_op_sparse(local_dims, idx, proj_op) # create the full projector operator for the n-th state of the idx-th party
    return projector_full


##################### Solution sheet 5 ####################


def Sx_sparse(N):
    """
    Returns the Sx operator for a system of `N` spin-1/2 particles as a sparse matrix.
    The Sx operator is defined as:
    Sx = (S+ + S-) / 2
    where S+ and S- are the raising and lowering operators, respectively.
    """
    
    S_plus_vec = np.sqrt((N - np.arange(0, N)) * (np.arange(0, N) + 1)) 
    S_plus = sparse.diags_array(S_plus_vec, offsets=-1)
    return (S_plus + S_plus.T) / 2

def Sy_sparse(N):
    """
    Returns the Sy operator for a system of `N` spin-1/2 particles as a sparse matrix.
    The Sy operator is defined as:
    Sy = (S+ - S-) / (2i)
    where S+ and S- are the raising and lowering operators, respectively.
    """
    
    S_plus_vec = np.sqrt((N - np.arange(0, N)) * (np.arange(0, N) + 1)) 
    S_plus = sparse.diags_array(S_plus_vec, offsets=-1)
    return (S_plus - S_plus.T) / (2j)

def Sz_sparse(N):
    """
    Returns the Sz operator for a system of `N` spin-1/2 particles as a sparse matrix.
    The Sz operator is defined as:
    Sz |m> = m |m>
    where m is the magnetic quantum number corresponding to the state |m>.
    """
    
    Sz_vec = np.arange(N + 1) - N / 2 # from -N/2 to N/2 in steps of 1 
    return sparse.diags_array(Sz_vec)

def build_spin_ops_sparse(N):
    """
    Returns the collective spin operators Sx, Sy, and Sz for a system of `N` spin-1/2 particles as sparse matrices.
    The collective spin operators are defined as:
    Sx = sum_i sigma_x^i / 2
    Sy = sum_i sigma_y^i / 2
    Sz = sum_i sigma_z^i / 2
    where sigma_x^i, sigma_y^i, and sigma_z^i are the single-site Pauli operators acting on the i-th particle.
    """

    Sx = Sx_sparse(N)
    Sy = Sy_sparse(N)
    Sz = Sz_sparse(N)
    return Sx, Sy, Sz

def _T_positive_parity_symm(N):
    """
    Returns the basis-change matrix from the full Dicke basis to the
    positive-parity symmetric subspace basis.
    """

    S = N / 2
    full_dim = N + 1
    L = int(np.floor(S)) + 1
    T = np.zeros((full_dim, L), dtype=float)
    for k in range(L):
        m = S - k
        i_pos = int(m + S)
        i_neg = int(-m + S)
        if abs(m) < 1e-12:
            T[i_pos, k] = 1.0
        else:
            T[i_pos, k] = 1.0 / np.sqrt(2.0)
            T[i_neg, k] = 1.0 / np.sqrt(2.0)
    return T

def Sx_symm(N):
    """
    Returns the Sx operator for a system of `N` spin-1/2 particles in the positive symmetric subspace as a sparse matrix.
    The Sx operator is defined as:
    Sx = (S+ + S-) / 2
    where S+ and S- are the raising and lowering operators, respectively.
    """

    T = _T_positive_parity_symm(N)

    Sx_full = Sx_sparse(N)
    Sx_full_arr = Sx_full.toarray() if hasattr(Sx_full, "toarray") else np.array(Sx_full)
    Sx_reduced = T.T @ (Sx_full_arr @ T)
    return sparse.csr_array(Sx_reduced)

def Sz2_symm(N):
    """
    Returns the Sz^2 operator for a system of `N` spin-1/2 particles in the positive symmetric subspace as a sparse matrix.
    The Sz^2 operator is defined as:
    Sz^2 |m> = m^2 |m>
    where m is the magnetic quantum number corresponding to the state |m>.
    Note that Sz vanishes in the positive symmetric subspace.
    """

    T = _T_positive_parity_symm(N)

    Sz_full = Sz_sparse(N)
    Sz_full_arr = Sz_full.toarray() if hasattr(Sz_full, "toarray") else np.array(Sz_full)
    Sz2_reduced = T.T @ ((Sz_full_arr @ Sz_full_arr) @ T)
    return sparse.csr_array(Sz2_reduced)

def sigma_x_sparse():
    """
    Returns the single-site sigma_x operator for a spin-1/2 particle as a sparse matrix.
    The sigma_x operator is defined as:
    sigma_x |0> = |1>
    sigma_x |1> = |0>
    """

    return sparse.csr_array([[0, 1], [1, 0]])

def sigma_y_sparse():
    """
    Returns the single-site sigma_y operator for a spin-1/2 particle as a sparse matrix.
    The sigma_y operator is defined as:
    sigma_y |0> = -i |1>
    sigma_y |1> = i |0>
    """

    return sparse.csr_array([[0, 1j], [-1j, 0]])

def sigma_z_sparse():
    """
    Returns the single-site sigma_z operator for a spin-1/2 particle as a sparse matrix.
    The sigma_z operator is defined as:
    sigma_z |0> = -|0>
    sigma_z |1> = |1>
    """

    return sparse.csr_array([[-1, 0], [0, 1]])
