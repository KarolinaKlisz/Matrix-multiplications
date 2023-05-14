#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def multiplication(matrix1, matrix2, l, k):
    if k>l:
        start_time = perf_counter()
        operations= strassen_multiplication(matrix1, matrix2)[1]
        duration = perf_counter() - start_time
    else:
        start_time = perf_counter()
        operations = tradictional_multiplication(matrix1, matrix2)
        duration = perf_counter() - start_time
    return (operations, duration)

def tradictional_multiplication(matrix1, matrix2): 
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])
    sum1 = 0
    counter1 = 0
    counter2 = 0

    if cols1 == rows2:  
        multiply = [[0 for j in range(cols2)] for i in range(rows1)]
        for i in range(rows1):
            for j in range(cols2):
                for k in range(cols1):
                    multiplication = matrix1[i][k] * matrix2[k][j]
                    counter1 += 1
                    sum1 = sum1 + multiplication
                    counter2 += 1
            multiply[i][j] = sum1
            sum1 = 0
    else:
        return ("The dimensions of the matrices are incorrect, they cannot be multiplied.")
    return counter1 + counter2 

def strassen_multiplication(matrix1, matrix2):
    rows1 = matrix1.shape[0]
    cols1 = matrix1.shape[1]
    rows2 = matrix2.shape[0]
    cols2 = matrix2.shape[1]

    if cols1 == rows1 == cols2 == rows2:
        n = rows1
        k = n // 2

        if n == 1:
            return matrix1 * matrix2, 1

        A11 = matrix1[:k, :k]
        A12 = matrix1[:k, k:]
        A21 = matrix1[k:, :k]
        A22 = matrix1[k:, k:]
        B11 = matrix2[:k, :k]
        B12 = matrix2[:k, k:]
        B21 = matrix2[k:, :k]
        B22 = matrix2[k:, k:]

        P1, fp = strassen_multiplication(A11 + A22, B11 + B22)
        P2, fp = strassen_multiplication(A21 + A22, B11)
        P3, fp = strassen_multiplication(A11, B12 - B22)
        P4, fp = strassen_multiplication(A22, B21 - B11)
        P5, fp = strassen_multiplication(A11 + A12, B22)
        P6, fp = strassen_multiplication(A21 - A11, B11 + B12)
        P7, fp = strassen_multiplication(A12 - A22, B21 + B22)

        fp = 7 * fp + 18 * k * k 

        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 - P2 + P3 + P6

        C = np.zeros(shape=(n, n))
        C[:k, :k] += C11
        C[:k, k:] += C12
        C[k:, :k] += C21
        C[k:, k:] += C22

        return C, fp

    else:
        return ("The dimensions of the matrices are incorrect, they cannot be multiplied.")

k_max=8 
l=3 

while l>2 and l<k_max:
    operations = []
    time = []
    size = []
    for k in range(2, k_max+1):
        A = np.random.randint(10, size=(2**k, 2**k))
        print(A.shape)
        size.append(2**k)
        B = np.random.randint(10, size=(2**k, 2**k))
        result = multiplication(A, B, l, k)
        operations.append(result[0])
        time.append(result[1])
    plt.figure(1)
    plt.plot(size, operations)
    plt.axvline(2**l, color='red', linestyle="--")
    plt.xlabel('Matrix size')
    plt.ylabel('Operations')
    plt.title(f"Number of operations for l={l}")
    plt.savefig(f'Operations{l}.png')
    plt.figure(2)
    plt.plot(size, time)
    plt.axvline(2 ** l, color='red', linestyle="--")
    plt.xlabel('Matrix size')
    plt.ylabel('Time')
    plt.title(f"Computation time for l={l}")
    plt.savefig(f'time{l}.png')
    plt.show()
    l+=2 


# In[ ]:




