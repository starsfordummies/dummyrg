import numpy as np 
import scipy.sparse.linalg as LA

from tensornetwork import ncon


from timeit import default_timer as timer
from datetime import timedelta

""" 
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
"""

chiL = 200
chiR = 220
dd = 2 
dMPO = 5

dimH = chiL*dd*dd*chiR

guessTheta = np.random.rand(chiL, dd, dd, chiR)

le = np.random.rand(chiL, dMPO, chiL)
re = np.random.rand(chiR, dMPO, chiR)
ww = np.random.rand(dMPO,dMPO,dd,dd)
ww = ww + ww.conj().transpose(1,0,3,2)

toleig = 1e-2
#theta = th.reshape(chiL, dd, dd, chiR)


def HthetaL(th: np.ndarray) -> np.ndarray:
    #print(np.shape(th))
    th = th.reshape(chiL,dd,dd,chiR)
    hth= ncon([le, ww, th, ww, re],
        [[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]]).reshape(dimH)
    #print(np.shape(hth))
    return hth

Heff = LA.LinearOperator((dimH,dimH), matvec=HthetaL)

start = timer()
lam0, eivec0 = LA.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig, ncv=3)
end = timer()
print(lam0,end-start) 
start = timer()
lam0, eivec0 = LA.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig, ncv=5)
end = timer()
print(lam0,end-start) 

start = timer()
lam0, eivec0 = LA.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig, ncv=10)
end = timer()
print(lam0,end-start) 

start = timer()
lam0, eivec0 = LA.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig, ncv=20)
end = timer()
print(lam0,end-start) 

start = timer()
lam0, eivec0 = LA.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig, ncv=30)
end = timer()
print(lam0,end-start) 

start = timer()
lam0, eivec0 = LA.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig, ncv=40)
end = timer()
print(lam0,end-start) 
