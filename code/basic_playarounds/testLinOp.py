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
chiR = 230
dd = 2 
dMPO = 5

dimH = chiL*dd*dd*chiR

guessTheta = np.random.rand(chiL, dd, dd, chiR)

le = np.random.rand(chiL, dMPO, chiL)
re = np.random.rand(chiR, dMPO, chiR)
ww = np.random.rand(dMPO,dMPO,dd,dd)
ww = ww + ww.conj().transpose(1,0,3,2)

toleig = 1e-3
#theta = th.reshape(chiL, dd, dd, chiR)


def HthetaL(th: np.ndarray) -> np.ndarray:
    #print(np.shape(th))
    th = th.reshape(chiL,dd,dd,chiR)
    hth= ncon([le, ww, th, ww, re],
        [[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]]).reshape(dimH)
    #print(np.shape(hth))
    return hth

Heff = LA.LinearOperator((dimH,dimH), matvec=HthetaL)

dts = []
for j in range(1,10):
    start = timer()
    lam0, eivec0 = LA.eigsh(Heff, k=1, which='SA', v0=guessTheta, tol=toleig)
    end = timer()
    dt = end-start
    dts.append(dt)

# Remove smallest and largest
print(dts)
dts.sort()
dts.pop(0)
dts.pop(-1)

print(np.average(dts))
"""
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
"""


o = [1,2,3,4,5,6,7]
def HthetaOrder(th: np.ndarray) -> np.ndarray:
    #print(np.shape(th))
    th = th.reshape(chiL,dd,dd,chiR)
    order = [[-1,o[0],o[1]],[o[0],o[2],-2,o[3]],[o[1],o[3],o[4],o[5]],[o[2],o[6],-3,o[4]],[-4,o[6],o[5]]]
    hth= ncon([le, ww, th, ww, re], order).reshape(dimH)
    #[[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]]
    #print(np.shape(hth))
    return hth

HeffO = LA.LinearOperator((dimH,dimH), matvec=HthetaOrder)


start = timer()
lam0, eivec0 = LA.eigsh(HeffO, k=1, which='SA', v0=guessTheta, tol=toleig)
end = timer()
print(lam0,end-start) 

o = [2,1,3,4,5,6,7]
def HthetaOrder(th: np.ndarray) -> np.ndarray:
    #print(np.shape(th))
    th = th.reshape(chiL,dd,dd,chiR)
    order = [[-1,o[0],o[1]],[o[0],o[2],-2,o[3]],[o[1],o[3],o[4],o[5]],[o[2],o[6],-3,o[4]],[-4,o[6],o[5]]]
    hth= ncon([le, ww, th, ww, re], order).reshape(dimH)
    #[[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]]
    #print(np.shape(hth))
    return hth

HeffO = LA.LinearOperator((dimH,dimH), matvec=HthetaOrder)


start = timer()
lam0, eivec0 = LA.eigsh(HeffO, k=1, which='SA', v0=guessTheta, tol=toleig)
end = timer()
print(lam0,end-start) 

o = [2,1,3,4,5,6,7]
def HthetaOrder(th: np.ndarray) -> np.ndarray:
    #print(np.shape(th))
    th = th.reshape(chiL,dd,dd,chiR)
    order = [[-1,o[0],o[1]],[o[0],o[2],-2,o[3]],[o[1],o[3],o[4],o[5]],[o[2],o[6],-3,o[4]],[-4,o[6],o[5]]]
    hth= ncon([le, ww, th, ww, re], order).reshape(dimH)
    #[[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]]
    #print(np.shape(hth))
    return hth

HeffO = LA.LinearOperator((dimH,dimH), matvec=HthetaOrder)


start = timer()
lam0, eivec0 = LA.eigsh(HeffO, k=1, which='SA',v0=guessTheta, tol=toleig)
end = timer()
print(lam0,end-start) 

o = [2,1,4,3,6,5,7]
def HthetaOrder(th: np.ndarray) -> np.ndarray:
    #print(np.shape(th))
    th = th.reshape(chiL,dd,dd,chiR)
    order = [[-1,o[0],o[1]],[o[0],o[2],-2,o[3]],[o[1],o[3],o[4],o[5]],[o[2],o[6],-3,o[4]],[-4,o[6],o[5]]]
    hth= ncon([le, ww, th, ww, re], order).reshape(dimH)
    #[[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]]
    #print(np.shape(hth))
    return hth

HeffO = LA.LinearOperator((dimH,dimH), matvec=HthetaOrder)


start = timer()
lam0, eivec0 = LA.eigsh(HeffO, k=1, which='SA',v0=guessTheta, tol=toleig)
end = timer()
print(lam0,end-start) 


# Now more seriously

def build_perms(n: int, perm: list, listPerms: list) -> list:
    
    outList = listPerms

    if n == 2:
      #print(f"doing last two with input {perm}")
      outList.append(perm)
      last = perm[:]
      last[-2], last[-1] = last[-1], last[-2]
      outList.append(last)

    else: # n > 2
      #print(perm)
      #print(f"Starting from {perm[-n:]}")
      for i, si in enumerate(perm[-n:]):
          #print(i, si)
          tperm = perm[:]
          tperm[-n], tperm[-n+i] = tperm[-n+i], tperm[-n]
          #print(f"Passing {tperm} to {n-1} buildperms")

          build_perms(n-1, tperm, outList)

          #print(f"output: {outList}")

    return outList
          

      #perm(tensor,n-1)

li1 = [1,2,3,4,5,6,7]
out = []
build_perms(len(li1), li1, out)


chiL = 220
chiR = 270
dd = 2 
dMPO = 5

dimH = chiL*dd*dd*chiR

guessTheta = np.random.rand(chiL, dd, dd, chiR)

le = np.random.rand(chiL, dMPO, chiL)
re = np.random.rand(chiR, dMPO, chiR)
ww = np.random.rand(dMPO,dMPO,dd,dd)
ww = ww + ww.conj().transpose(1,0,3,2)

toleig = 1e-3
dtmin = 1000

for o in out:
    def HthetaOrder(th: np.ndarray) -> np.ndarray:
        #print(np.shape(th))
        th = th.reshape(chiL,dd,dd,chiR)
        order = [[-1,o[0],o[1]],[o[0],o[2],-2,o[3]],[o[1],o[3],o[4],o[5]],[o[2],o[6],-3,o[4]],[-4,o[6],o[5]]]
        hth= ncon([le, ww, th, ww, re], order).reshape(dimH)
        #[[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]]
        #print(np.shape(hth))
        return hth

    HeffO = LA.LinearOperator((dimH,dimH), matvec=HthetaOrder)

    dts = []
    for i in range(0,8):
            
        start = timer()
        lam0, eivec0 = LA.eigsh(HeffO, k=1, which='SA',v0=guessTheta, tol=toleig)
        end = timer()
        dt = end-start
        dts.append(dt)

    # Remove smallest and largest
    print(dts)
    dts.sort()
    dts.pop(0)
    dts.pop(-1)

    print(o, np.average(dts))

    if np.average(dts) < dtmin:
        dtmin = np.average(dts)
        omin = o
    
    print("min::")
    print(omin)
    print(dtmin)
