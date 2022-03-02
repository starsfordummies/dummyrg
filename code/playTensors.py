import numpy as np
from numpy import linalg as LA
from ncon import ncon

# initialize complex random tensor
E = np.random.rand(2,3,4) + 1j*np.random.rand(2,3,4)

print(E)


##### Ex.1.2(a):Transpose
A = np.random.rand(4,4,4,4)
Atilda = A.transpose(3,0,1,2)

##### Ex.1.2(b):Reshape
B = np.random.rand(4,4,4)
Btilda = B.reshape(4,4**2)

##### Ex.1.5(b): Contraction using ncon
d = 10
A = np.random.rand(d,d,d); B = np.random.rand(d,d,d,d)
C = np.random.rand(d,d,d); D = np.random.rand(d,d)

TensorArray = [A,B,C,D]
IndexArray = [[1,-2,2],[-1,1,3,4],[5,3,2],[4,5]]

E = ncon(TensorArray,IndexArray,con_order = [5,3,4,1,2])


##### Ex2.2(b): SVD of tensor
d = 10; A = np.random.rand(d,d,d)
Am = A.reshape(d**2,d)
Um,Sm,Vh = LA.svd(Am,full_matrices=False)
U = Um.reshape(d,d,d); S = np.diag(Sm)
# check result
Af = ncon([U,S,Vh],[[-1,-2,1],[1,2],[2,-3]])
dA = LA.norm(Af-A)


"""
the SVD is also useful for generating random unitary and isometric tensors as shown here 
"""
##### Initialize unitaries and isometries
d1 = 10; d2 = 6;

# d1-by-d1 random unitary matrix U
U,_,_ = LA.svd(np.random.rand(d1,d1))
# d1-by-d2 random isometric matrix W
A = np.random.rand(d1,d2);
W,_,_ = LA.svd(A,full_matrices=False)

""" Restricted rank approx """

##### Ex2.4(a): SVD
d = 10; A = np.random.rand(d,d,d,d,d)
Um,S,Vhm = LA.svd(A.reshape(d**3,d**2),full_matrices=False)
U = Um.reshape(d,d,d,d**2)
Vh = Vhm.reshape(d**2,d,d)
##### truncation
chi = 8;
Vhtilda = Vh[:chi,:,:]
Stilda = np.diag(S[:chi])
Utilda = U[:,:,:,:chi]
B = ncon([Utilda,Stilda,Vhtilda],[[-1,-2,-3,1],[1,2],[2,-4,-5]])
##### compare
epsAB = LA.norm(A-B) / LA.norm(A)
