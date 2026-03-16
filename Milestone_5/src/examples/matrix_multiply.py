# matrix_multiply.py
# Example program for CodeXcelerate transpilation
# Algorithm: Matrix Multiplication — O(n³) — extreme Python slowness
# Expected speedup after transpilation: 1000–8000x

import random

def create_matrix(rows, cols, seed=42):
    random.seed(seed)
    return [[random.uniform(0, 1) for _ in range(cols)] for _ in range(rows)]

def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

SIZE = 200
A = create_matrix(SIZE, SIZE, seed=42)
B = create_matrix(SIZE, SIZE, seed=123)

C = matrix_multiply(A, B)
print(f"Multiplied {SIZE}x{SIZE} matrices")
print(f"Result[0][0] = {C[0][0]:.6f}")
print(f"Result[0][1] = {C[0][1]:.6f}")
