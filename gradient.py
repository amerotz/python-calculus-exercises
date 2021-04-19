import numpy as np
import matplotlib.pyplot as plt

print("How do we optimize numerically a function?")
print("We start from an initial x0 and then take a \"step\" in descent direction.")
print("The direction of steepest descent is given by:\n")
print("\t- grad(f(x0))\n\t_____________\n\t|grad(f(x0))|\n")
print("From here we have a succession of x_k such that:\n")
print("\tx_k+1 = x_k + a*p_k\n")
print("where p_k is a vector in the direction of steepest descent and a is the length of the step.")
print("However this criteria isn't sufficient to ensure that we'll eventually converge to a minimum in f.")
print("We have to choose the right length of the step we take in the direction p_k.")
print("A step too long may overcome the point we're looking for. So:\n")
print("\tx_k+1 = x_k + b1*a*p_k\n")
print("where b1 is a factor by which we scale our step.\n")
print("We must also ensure that the step isn't too short. To achieve this:\n")
print("\tgrad(f(x_k+a*p_k)) >= b2*grad(f(x_K))\n")
print("which means that the gradient between two iterates should vary of at least of b2 percent.\n")
print("These conditions are known as Wolfe condition. The first one is also known as Armijo condition.")
print("To simplify the problem, we'll only consider convex functions, for which a point of local minimum is always a point of global minimum too")
print("We know present a version of this algorithm with fixed step length.")

def gradientDescent(der, x0, maxIt, a, tol):
    err = []
    err.append(x0)
    x = x0
    it = 0
    f_ = der(x)
    p = - f_ 
    
    while np.abs(f_) > tol and it < maxIt:
        x += a*p
        err.append(x)
        f_ = der(x)
        if f_ != 0:
            p = - f_ 
        it += 1
        
    err = np.array(err)
    xvec = np.copy(err)
    for i in range(err.size):
      err[i] = np.linalg.norm(err[i] - x)
    return x, it, err, xvec

f = lambda x : x**2
der = lambda x, h=10**-10 : (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h))/(12*h)

x0 = 2
maxIt = 100
tol = 10**-10
a = 1

x, it, err, xvec = gradientDescent(der, x0, maxIt, a, tol)
print("x = ", x, "\tIterations: ", it)
print("F(x) = ", f(x))

graphX = np.linspace(x-1, x+1, 100)
plt.figure(0)
plt.title("Function")
plt.plot(graphX, f(graphX))
plt.grid(True)
plt.show()

plt.figure(1)
plt.title("Original Function and iterates (a = " + str(a) +")")
plt.grid(True)

graphX = np.linspace(min(xvec), max(xvec), 100)
fx = f(graphX)
plt.plot(graphX, fx, label="f(x)")
plt.scatter(xvec, f(xvec), label="Algorithm", marker="o", color="red")
plt.legend(loc="best")
plt.show()

plt.figure(2)
plt.title("Gradient descent fixed step error (tol = " + str(tol) + ")")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.grid(True)
plt.plot(err)
plt.show()

def gradientBT(f, der, x0, maxIt, tol):
    err = []
    err.append(x0)
    alphas = []
    x = x0
    it = 0
    f_ = der(x)
    p = - f_ 
    print(f_)
    print(p)
    
    c = 0.25
    rho = 0.5
    
    while np.abs(f_) > tol and it < maxIt:
        a = 1
        while f(x + a*p) > f(x) + a*c*p*f_ and np.abs(a) > 1e-5:
            a *= rho
        alphas.append(a)
        x += a*p
        err.append(x)
        f_ = der(x)
        if f_ != 0:
            p = - f_ 
        it += 1
        
    err = np.array(err)
    xvec = np.copy(err)
    alphas = np.array(alphas)
    for i in range(err.size):
      err[i] = np.linalg.norm(err[i] - x)
    return x, it, err, xvec, alphas

x, it, xvec, err, alphas = gradientBT(f, der, x0, maxIt, tol)
print("x = ", x, "\tIterations: ", it)
print("F(x) = ", der(x))

plt.figure(3)
plt.title("Original Function and iterates (backtracking)")
plt.grid(True)

fx = f(graphX)
plt.plot(graphX, fx, label="f(x)")
plt.scatter(xvec, f(xvec), label="Algorithm", marker="o", color="red")
plt.legend(loc="best")
plt.show()

plt.figure(4)
plt.title("Gradient descent backtraking error (tol = " + str(tol) + ")")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.grid(True)
plt.plot(err)
plt.show()

plt.figure(5)
plt.title("Step lenght variation")
plt.xlabel("Iterations")
plt.ylabel("Alpha")
plt.grid(True)
plt.plot(alphas)
plt.show()


f = lambda x : 0.5*x[0]**2 + 4.5*x[1]**2
der = lambda x : np.array([x[0], 9*x[1]])
x0 = [-1, 1]

def gradientBT2(der, x0, maxIt, tol):
    err = []
    err.append(x0)
    alphas = []
    x = x0
    it = 0
    f_ = der(x)
    p = - f_ 
    
    c = 0.25
    rho = 0.5
    
    while np.abs(np.all(f_)) > tol and it < maxIt:
        a = 1
        dotpf = np.dot(p, f_)
        fx = f(x)
        while f(x + a*p) > fx + a*c*dotpf and np.abs(a) > 1e-5:
            a *= rho
        alphas.append(a)
        x += a*p
        err.append(x)
        f_ = der(x)
        p = - f_ 
        it += 1
        
    err = np.array(err)
    xvec = np.copy(err)
    alphas = np.array(alphas)
    """
    print(err)
    for i in range(err.size):
      err[i, :] = np.linalg.norm(err[i, :] - x, 2)
      """
    return x, it, xvec, alphas

x, it, xvec, alphas = gradientBT2(der, x0, maxIt, tol)
print("x = ", x, "\tIterations: ", it)
print("F(x) = ", der(x))

plt.figure(6)
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f([X[:],Y[:]])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=45)
ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)
plt.show()