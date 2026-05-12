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