cimport scipy.linalg.cython_blas as blas

def compute_correlation(py_trans_a, py_trans_b, py_m, py_n, py_k, py_alpha, py_a, py_lda,
          py_b, py_ldb, py_beta, py_c, py_ldc):
    cdef bytes by_trans_a=py_trans_a.encode()
    cdef bytes by_trans_b=py_trans_b.encode()
    cdef char* trans_a = by_trans_a
    cdef char* trans_b = by_trans_b
    cdef int M, N, K, lda, ldb, ldc
    M = py_m
    N = py_n
    K = py_k
    lda = py_lda
    ldb = py_ldb
    ldc = py_ldc
    cdef float alpha, beta
    alpha = py_alpha
    beta = py_beta
    cdef float[:, ::1] A
    A = py_a
    cdef float[:, ::1] B
    B = py_b
    cdef float[:, ::1] C
    C = py_c
    blas.sgemm(trans_a, trans_b, &M, &N, &K, &alpha, &A[0, 0], &lda,
               &B[0, 0], &ldb, &beta, &C[0, 0], &ldc)
