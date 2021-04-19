import numpy as np
import matplotlib.pyplot as plt

print("We know how to solve a linear system Ax = b when the number of xs is equal to the number of equations.")
print("What do we do if there are more equations than variables?")
print("We solve the linear least squares problem (LSQ):\n")
print("\t min ||Ax - b||^2 = min ||r||^2\n")
print("where || . || denotes 2-norm.\nWe minimize the squared norm of the difference between Ax and b.\n")
print("We have different options depending on k = rank(A), (A is a m*n matrix):\n")
print("\tk = n -> unique solution")
print("\tk < n -> infinite solutions\n")

print("CASE k = n\n")

print("To minimize the norm ||Ax - b||^2 means to minimize:\n\n\tf(x) = xT AT Ax - 2xT Ab + bT b\n")
print("This happens when grad(f) = 0, i.e. when:\n\n\t2AT Ax - AT b = 0\n")
print("This allows us to solve the system AT Ax = AT b with the methods we explained before.")
print("The LSQ problem allows us to calculate an interpolator for a given dataset.\n")


print("\nTheese tools allow us to extrapolate from a dataset the function that produced it.\n")

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])

print(x)
print(y)

def LSQapprox(x, y, n):
    m = x.size
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            A[i, j] = x[i]**j
    return A

def poly(alpha, x0):
    x = 0
    for i in range(alpha.size):
        x += alpha[i]*(x0**i)
    return x

A1 = LSQapprox(x, y, 5)
A = A1.T @ A1
b = A1.T @ y

alpha = np.linalg.solve(A, b)

plt.figure(0)
plt.title("Interpolation via LSQ with normal equations")
plt.plot(x, y, label="Sampled data")
graphX = np.linspace(min(x), max(x), 10*x.size)
plt.plot(graphX, poly(alpha, graphX), label = "Interpolated data",)
plt.legend(loc="best")
plt.show()

print("\nCASE k <= n\n")
print("In this case we can minimize the norm using singular value decomposition (SVD).")
print("Given A, a m*n matrix, we can decompose it in three matrixes U, S and V such that:\n")
print("\tU is a m*m orthogonal matrix\n\tS is a diagonal matrix containing singular values\n\tV is a n*n orthogonal matrix\n")
print("and A = U S VT.\n")

from scipy import linalg

U, S, Vt = linalg.svd(A)

def svdsolve(U, S, Vt, b):
    V = Vt.T
    x = np.zeros(b.size)
    x[:] = V[:,] @ ((U.T[:,] @ b) / S[:])
    #x = V @ ((U.T @ b) / S)
    return x

alpha = svdsolve(U, S, Vt, b)
plt.figure(1)
plt.title("Interpolation via LSQ with SVD")
plt.plot(x, y, label="Sampled data")
plt.plot(graphX, poly(alpha, graphX), label = "Interpolated data",)
plt.legend(loc="best")
plt.show()

print("SVD can be used to compress images too.")
print("Given A, a matrix representing an image, we can approximate it with another matrix Ai such that:\n")
print("\trank(Ai) <= rank(A) and Ai = sum for j=0 to i si * ui * vTi")

from matplotlib.image import imread

A = np.mean(imread("kitten.jpg"), -1)

U, S, Vt = linalg.svd(A, full_matrices=False)
S = np.diag(S)

def imgSVD(U, S, Vt, p):
    A = U[:,:p] @ S[:p,:p] @ Vt[:p, :]
    return A

plt.imshow(A, cmap ="gray")
plt.show()

err = np.zeros(4)
num = np.zeros(4)

j = 0
for i in range(0, 21, 5):
    plt.figure(j+2)
    A1 = imgSVD(U, S, Vt, i)
    err[j-1] = np.linalg.norm(A-A1, 2)/np.linalg.norm(A, 2)
    num[j-1] = i
    plt.title(str(i) + " order approx")
    plt.imshow(A1, cmap ="gray")
    plt.axis("off")
    plt.show()
    j += 1


plt.figure(6)
plt.plot(num, err)
plt.show()

j = 0
for i in range(100, 50, -10):
    plt.figure(j+2)
    A1 = imgSVD(U, S, Vt, i)
    err[j-1] = np.linalg.norm(A-A1, 2)/np.linalg.norm(A, 2)
    num[j-1] = i
    plt.title(str(i) + " order approx")
    plt.imshow(A1, cmap ="gray")
    plt.axis("off")
    plt.show()
    j += 1


plt.figure(6)
plt.plot(num, err)
plt.show()




