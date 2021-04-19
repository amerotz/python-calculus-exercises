import numpy as np
from numpy import pi as PI
import matplotlib.pyplot as plt

print("Given a dataset of nodes (x, y), how do we find f such that f(x) = y?")
print("We have to build an interpolator for y, which will be our f.")
print("We'll use Lagrange's polynomial interpolator.\n")

print("The function we'll interpolate is Runge's function:\n\n\tf(x) = 1 / (1 + x^2)\n")

runge = lambda x : 1 / (1 + x*x)

sampleN = 11
start = -5
stop = 5
print("We'll sample it over", sampleN, "equally spaced points in [", start, ",", stop, "].\n")

x = np.linspace(start, stop, sampleN)
print("x = ", x)
y = runge(x)
print("y = ", y)

print("\nFrom here, we can calculate the interpolator and plot the results.\n")

plt.figure(0)
plt.title("Runge's function Lagrange interpolation")
graphX = np.linspace(start, stop, 10*sampleN)
plt.plot(graphX, runge(graphX), label="Runge's f", linewidth=2)
plt.plot(x, y, color="red", label = "Sampled f", marker=".", linewidth=1)

def lagrange(x, y, x0):
    n = y.size
    px = 0
    for k in range(n):
        fk = 1
        for j in range(n):
            if j != k:
                fk *= (x0 - x[j]) / (x[k] - x[j])
        px += y[k]*fk
    return px

plt.plot(graphX, lagrange(x, y, graphX), label="Interpolated f")
plt.legend(loc="best")
plt.show()

print("The overall error you see derives from the little number of points we used to sample f.")
print("However the big spikes at the bounds of the interval are what it's truly concerning.\n")
print("To avoid this phenomenon, we may use a different set of points known as Chebyshev-Gauss-Lobatto's points.\n")

def chebgausslob(start, stop, n):
    x = np.zeros(n)
    for i in range(n):
        xhat = -np.cos(PI*i/(n-1))
        x[i] = (start + stop)/2 + xhat*(stop - start)/2
    return x

x = chebgausslob(start, stop, sampleN)
print("x = ", x)
y = runge(x)
print("y = ", y)

plt.figure(1)
plt.title("Runge's function Lagrange interpolation (CGL points)")
plt.plot(graphX, runge(graphX), label="Runge's f", linewidth=2)
plt.plot(x, y, color="red", label = "Sampled f", marker=".", linewidth=1)
plt.plot(graphX, lagrange(x, y, graphX), label="Interpolated f")
plt.legend(loc="best")
plt.show()

print("\nAnother slightly different set of point is Chebyshev-Gauss' points.\n")

def chebgauss(start, stop, n):
    x = np.zeros(n)
    for i in range(n):
        xhat = -np.cos( ((2*i + 1) / n) * PI/2)
        x[i] = (start + stop)/2 + xhat*(stop - start)/2
    return x

x = chebgauss(start, stop, sampleN)
print("x = ", x)
y = runge(x)
print("y = ", y)

plt.figure(1)
plt.title("Runge's function Lagrange interpolation (CG points)")
plt.plot(graphX, runge(graphX), label="Runge's f", linewidth=2)
plt.plot(x, y, color="red", label = "Sampled f", marker=".", linewidth=1)
plt.plot(graphX, lagrange(x, y, graphX), label="Interpolated f")
plt.legend(loc="best")
plt.show()

print("\nTheese tools allow us to extrapolate from a dataset the function that produced it.\n")

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])

print(x)
print(y)

plt.figure(2)
plt.plot(x, y, label="Sampled data", marker = "o")
graphX = np.linspace(min(x), max(x), 10*x.size)
plt.plot(graphX, lagrange(x, y, graphX), label = "Interpolated data")
plt.legend(loc="best")
plt.show()
