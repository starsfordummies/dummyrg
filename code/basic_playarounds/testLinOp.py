import numpy as np 
import scipy.sparse.linalg as LA

from tensornetwork import ncon

def mv(v):
    return np.array([2*v[0], 3*v[1]])

A = LA.LinearOperator((2,2), matvec=mv)

A.matvec(np.ones(2))

eigenvalues, eigenvectors = LA.eigsh(A, k=1, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal')

print(eigenvalues)
print()
print(eigenvectors)

# Can I do it with tensors? eg. d1xd2 tensors

d1 = 2 
d2 = 3

WW = np.random.rand(d1,d2,d1,d2)

WWr = WW.reshape(d1*d2,d1*d2)

def WW_times_theta(theta): 
    return WWr @ theta
    #return ncon([WW,theta],[[-1,-2,1,2],[1,2]])

Wlo = LA.LinearOperator((d1*d2,d1*d2), matvec = WW_times_theta)


eigenvalues, eigenvectors = LA.eigsh(Wlo, k=1, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal')

print("Eigs")
print(eigenvalues)
print()
print(eigenvectors)
