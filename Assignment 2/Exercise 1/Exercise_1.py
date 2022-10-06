import torch
import math
import random


def matrix_mult(A, B):
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])

    # Assume that we are always passing them in the correct order. No swapping.
    if cols_a != rows_b:
        raise Exception('Incompatible matrices.')

    C = [[0]*cols_b]*rows_a
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(rows_b):
                C[i][j] += A[i][k] * B[k][j]

    return C


if __name__ == '__main__':
    A = [[1, 2, 3],
         [1, 2, 3]]
    B = [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]

    print(matrix_mult(A, B))