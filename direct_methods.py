import numpy as np

print("Given a linear system Ax = b, how do we find x?\n")
print("It depends on the type of matrix\n")
print("Given b:\n")

b = np.array([4, 3, 0, 4])
print(b, "\n")
print("If A is an upper triangular matrix:\n")

A1 = np.array([[1, 0, 1, 0],
              [0, 2, 4, 2],
              [0, 0, 2, 1],
              [0, 0, 0, 1]])
print(A1, "\n")
print("We solve for x via sostitution:\n")

def UTriSol(A, b):
  n = b.size
  x = np.zeros(n) #empty solution
  x[n-1] = b[n-1] / A[n-1, n-1]; #last x
  for i in range(n-1, -1, -1):
      s = 0 #sum of previous xs
      for j in range(i+1, n):
          s += A[i, j] * x[j]
          
      x[i] = (b[i] - s)/A[i, i]   
      
  return x

x = UTriSol(A1, b)
print("x = ", x, "\n")

print("The same applies if A is a lower triangular matrix:\n\n")

A2 = np.array([[1, 0, 0, 0],
              [0, 2, 0, 0],
              [1, 4, 2, 0],
              [0, 2, 1, 1]])

print(A2, "\n")

print("We do the same, but in reverse:\n")

def LTriSol(A, b):
    n = b.size
    x = np.zeros(n) #empty solution
    x[0] = b[0] / A[0, 0] #first x
    for i in range(1, n):
        s = 0
        for j in range(0, i):
            s += A[i, j] * x[j]
        
        x[i] = (b[i] - s)/A[i, i]
    
    return x

x = LTriSol(A2, b)
print("x = ", x, "\n")

print("What if A is not a triangular matrix?")

A3=np.array([[3,-1,1,-2],[0,2,5,-1],[1,0,-7,1],[0,2,1,1]],dtype=np.float)
A3=np.matmul(A3,np.transpose(A3))

print(A3, "\n")

print("We can decompose it in a L triangular and an U triangular matrix:\n")

def LU(A):
    a = np.copy(A)
    n = a.shape[1]
    
    for k in range(n):
        if a[k, k] != 0:
            a[k+1 :, k] = a[k+1 :, k] / a[k, k]
            a1 = np.expand_dims(a[k+1:, k], 1)
            a2 = np.expand_dims(a[k, k+1:], 0)
            a[k+1:, k+1:] = a[k+1:, k+1:] - (a1 * a2)
    return a

lu = LU(A3)
print(lu, "\n")

print("This matrix contains both matrixes in one. The main diagonal of the lower will always contain 1s.\n")
print("From here we have:\n")
print("\tA = LU and Ax = b\n")
print("This means that:\n")
print("\tLUx = b\n")
print("Let y = Ux, then:\n")
print("\tLy = b\n")
print("We can easily solve for y beacuse L is a lower triangular matrix.\n")

L = np.tril(lu, -1) + np.eye(4, 4)
print(L, "\n")
y = LTriSol(L, b)
print("y = ", y, "\n")

print("We then have:\n")
print("\tUx = y\n")
print("We finally solve for x, being U an upper triangular matrix.\n")

U = np.triu(lu, 0)
print(U, "\n")
x = UTriSol(U, y)
print("x = ", x, "\n")

print("This method requires the calculation of two matrices in time O(n^3/3).")
print("However, such a decomposition isn't always possible.")
print("A variant of this method, called Lu factorization with partial pivoting, solves this problem, as long as A is non singular.")
print("Moreover, this new version is stabler than the previous.\n")

print("Another decomposition method, roughly twice more efficient than Lu, is Cholesky factorization.")
print("Every symmetric and positive definite matrix can be written as:\n")
print("\tA = L * L^T\n")
print("This algorithm calculates such matrix:\n")

def cholesky(A):
    n = A.shape[1]
    L = np.zeros(A.shape)
    
    for i in range(n):
        for j in range(i):
            
            s1 = 0
            for k in range(j):
                s1 += L[j, k] * L[i, k]
                
            L[i, j] = (A[j, i] - s1) / L[j, j]
            
        s2 = 0
        for j in range(i):
            s2 += L[i, j] * L[i, j]
            
        if (A[i, i] - s2 < 0):
            return "Matrix is not positive definite"
        
        L[i, i] = np.sqrt(A[i, i] - s2)
        
    return L
     

L = cholesky(A3)
print(L, "\n")

print("We then apply the same substitution method as above:\n")

y = LTriSol(L, b)
print("\tLy = b;\t\ty = ", y, "\n")

x = UTriSol(L.T, y)
print("\tL^Tx = y;\tx = ", x, "\n")

print("The solution is computed in time O(n^3/6).\n")

print("To prove that theese methods work, here's the solution to Ax = b calculated using np.linalg.solve(A, b):\n")
x = np.linalg.solve(A3, b)
print("\tx = ", x)