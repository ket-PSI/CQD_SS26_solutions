import Comp_Quant_Dynam as cqd
import numpy as np   # standard numerics library
import math
from scipy.special import hermite as herm
import scipy.sparse as sparse # routines for sparse matrices

def single_spin_op(N):
    sx_list = []
    sy_list = []
    sz_list = []

    for i in range(N):

        sx_i = sparse.csr_matrix([[1]], dtype=complex)
        sy_i = sparse.csr_matrix([[1]], dtype=complex)
        sz_i = sparse.csr_matrix([[1]], dtype=complex)

        for j in range(N):

            sx_i = kron(sx_i, s_x if j == i else I, format='csr')
            sy_i = kron(sy_i, s_y if j == i else I, format='csr')
            sz_i = kron(sz_i, s_z if j == i else I, format='csr')

        sx_list.append(sx_i)    #list with all possible i values
        sy_list.append(sy_i)
        sz_list.append(sz_i) 

    return sx_list, sy_list, sz_list

sx_list, sy_list, sz_list = single_spin_op(N)