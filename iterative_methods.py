import numpy as np
import matplotlib.pyplot as plt

print("To solve the linear system Ax = b we may use iterative methods.")
print("This kind of method works with an initial arbitrary solution, iterating on which we approximate the true solution.")
print("Some of such methods rewrite A in this way:\n")
print("\tA = M - N\n")
print("where M is non singular and relatively easy to solve.")
print("Our system now looks like this:\n")
print("\t(M - N)x = b;\n\tMx - Nx = b;\n\tMx = Nx + b\n\tx = M'Nx + M'b\n")
print("Let T = M'N and c = M'b. We obtain:\n\n\tx = Tx + c\n")
print("We call T the method's iteration matrix.")
print("How are the two matrixes built?\n")
print("Given A, we can decompose it as:\n\n\tA = D - E - F\n\nwhere:\n")
print("\tD is A's main diagonal\n\t-E is the strictly lower triangular part\n\t-F is the strictly upper triangular part\n")

print("We'll test theese algorithms on the following matrix and column vector:\n")
n = 4
A = 5 * np.eye(n) + np.eye(n, n, 1) + np.eye(n, n, -1)
xTrue = np.ones(n)
b = np.dot(A, xTrue)
print(A, " = A\n\n", b, " = b\n")

print("They were specifically built to have\n\n\tx = ", xTrue, "\n\nas solution.\n")

print("We'll first consider Jacobi's method.\nIn this case, the iteration matrix J looks like this:\n")
print("\tM = D;\tN = E + F\n\tJ = D'(E + F) = I - D'A\n")
print("We'll feed the algorithm the matrix, the column vector and arbitrary solution of all zeroes.")
print("We'll also give it a maximum number of iterations, the true solution and a tolerance under which to stop.\n")

tol = 10**-6

def jacobi(A, b, x0, maxIt, xTrue, tol):
    n = x0.size
    it = 0
    vecs = np.zeros((maxIt,1))
    x = np.copy(x0)
    norm = tol+1
    while it < maxIt and norm > tol:
        xold = np.copy(x)
        vecs[it] = np.linalg.norm(xTrue - x)
        for i in range(n):
            s1 = 0
            for j in range(i):
                s1 += A[i, j] * xold[j]
            s2 = 0
            for j in range(i+1, n):
                s2 += A[i, j] * xold[j]
            x[i] = (b[i] - s1 -s2) / A[i, i]
        norm = np.linalg.norm(x-xold)/np.linalg.norm(x)
        it += 1
    return x, vecs, it

x, vecs1, it = jacobi(A, b, np.zeros(n), 25, xTrue, tol)
print("\tx = ", x, "\n\t(", it, " iterations, tol = ", tol, ")\n")
print("You see that our solution isn't perfect. But if we lower the tolerance we have:\n")
tol = 10**-10
x, vecs, it = jacobi(A, b, np.zeros(n), 25, xTrue, tol)
print("\tx = ", x, "\n\t(", it, " iterations, tol = ", tol, ")\n")

print("The second method we'll consider is Gauss-Seidels'.")
print("In this case, M = D - E and N = F. The iteration matrix G is equal to (D -E)'F.\n")
print("Proceeding as above, we have:\n")

tol = 10**-6

def gaussseidel(A, b, x0, maxIt, xTrue, tol):
    n = x0.size
    it = 0
    vecs = np.zeros((maxIt,1))
    x = np.copy(x0)
    norm = tol+1
    while it < maxIt and norm > tol:
        xold = np.copy(x)
        vecs[it] = np.linalg.norm(xTrue - x)
        for i in range(n):
            s1 = 0
            for j in range(i):
                s1 += A[i, j] * x[j]
            s2 = 0
            for j in range(i+1, n):
                s2 += A[i, j] * xold[j]
            x[i] = (b[i] - s1 -s2) / A[i, i]
        norm = np.linalg.norm(x-xold)/np.linalg.norm(x)
        it += 1
    return x, vecs, it

x, vecs2, it = gaussseidel(A, b, np.zeros(n), 25, xTrue, tol)
print("\tx = ", x, "\n\t(", it, " iterations, tol = ", tol, ")\n")
print("And with a lower tolerance:\n")
tol = 10**-10
x, vecs, it = gaussseidel(A, b, np.zeros(n), 25, xTrue, tol)
print("\tx = ", x, "\n\t(", it, " iterations, tol = ", tol, ")\n")

print("You'll notice that in this case this algorithm is almost twice faster than the former.")
print("This emerges clearly if we plot the error ||x - x*|| per iteration.")


plt.plot(vecs1, label='Gauss-Seidel', color='red', linewidth=1, marker='o')
plt.plot(vecs2, label='Jacobi', color='blue', linewidth=1, marker='.')

