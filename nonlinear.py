import numpy as np
import matplotlib.pyplot as plt

print("How do we solve nonlinear equations in the form F(x) = 0?")
print("The best way is through iterative methods.")
print("Given x_k, we have x_k+1 = G(x_k). This succession should converge to the solution x* for k -> +inf.")
print("However, this generally happens only for adequate values of the initial x0.\n")
print("The first method we'll analyze is the bisection method.\n")

def bisectSolve(f, a, b, maxIt, tol):
    err = []
    it = 0
    x = a + (b-a)*0.5
    val = f(x)
    err.append(x)
    tooSmall = abs(b-a) < tol + np.finfo(float).eps * max(abs(a), abs(b))
    while np.abs(val) > tol and it < maxIt and not tooSmall:
        if val < 0:
            a = x
        elif val > 0:
            b = x
        x = a + (b-a)*0.5
        val = f(x)
        err.append(x)
        tooSmall = abs(b-a) < tol + np.finfo(float).eps * max(abs(a), abs(b))
        it += 1
    err = np.array(err)
    xvec = np.copy(err)
    for i in range(err.size):
      err[i] = np.linalg.norm(err[i] - x)
    return x, it, err, xvec

fun = lambda x : x * np.sin(x)

a = 2.5
b = 3.5
maxIt = 100
tol = 10**-15
err = []
x, it, err, xvec = bisectSolve(fun, b, a, maxIt, tol)
print("x = ", x, "\tIteratons: ", it)
print("F(x) = ", fun(x))

plt.figure(0)
plt.title("Original Function (Bisection)")
graphX = np.linspace(a, b, 100)
plt.grid(True)

plt.plot(graphX, fun(graphX))
plt.plot([a, b], [0, 0], color = "black")
plt.show()

plt.figure(1)
plt.title("Bisection method error (tol = " + str(tol) + ")")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.grid(True)
plt.plot(err)
plt.show()

print("\nThe second method is the fixed point method.\n")

def fixedPoint(f, ph, maxIt, tol, x0, tol2 = np.finfo(float).eps):
    it = 0
    err = []
    err.append(x0)
    x = x0
    tooSmall = False
    while np.abs(f(x)) > tol and it < maxIt and not tooSmall:
        oldx = x
        x = x - f(x)*ph(x)
        err.append(x)
        tooSmall = np.abs(x - oldx) < tol2
        it += 1
    
    err = np.array(err)
    xvec = np.copy(err)
    for i in range(err.size):
      err[i] = np.linalg.norm(err[i] - x)
    return x, it, err, xvec

phi = lambda x : 1
x0 = 1

x, it, err, xvec = fixedPoint(fun, phi, maxIt, tol, x0)
print("x = ", x, "\tIteratons: ", it)
print("F(x) = ", fun(x))

plt.figure(2)
plt.title("Original Function (Fixed Point)")
plt.grid(True)

fx = fun(graphX)
plt.plot(graphX, fx, label="f(x)")
plt.plot(graphX, graphX - phi(graphX)*fx, label="x - f(x)*phi(x)")
plt.plot([a,b], [a, b], label="x")
plt.legend(loc="best")
plt.plot([a, b], [0, 0], color = "black")
plt.show()

plt.figure(3)
plt.title("Fixed point method error (tol = " + str(tol) + ")")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.grid(True)
plt.plot(err)
plt.show()

print("Another istance of this method is Newton's fixed point method.")

der = lambda x, h, f : (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h))/(12*h)
newt_phi = lambda x : 1/der(x, 10**-6, fun)

x, it, err, xvec = fixedPoint(fun, newt_phi, maxIt, tol, x0)
print("x = ", x, "\tIteratons: ", it)
print("F(x) = ", fun(x))

plt.figure(4)
plt.title("Original function (Newton FP)")
plt.grid(True)

fx = fun(graphX)
plt.plot(graphX, fx, label="f(x)")
plt.plot(graphX, graphX - newt_phi(graphX)*fx, label="x - f(x)/f'(x)")
plt.plot([a,b], [a, b], label="x")
plt.legend(loc="best")
plt.plot([a, b], [0, 0], color = "black")
plt.show()

plt.figure(5)
plt.title("Newton FP method error (tol = " + str(tol) + ")")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.grid(True)
plt.plot(err)
plt.show()