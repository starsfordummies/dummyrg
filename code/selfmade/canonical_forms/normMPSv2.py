import numpy as np
from ncon import ncon

# Physical dimension
d = 2
# Bond dimension
chi = 5


""" 
Try to make it for a larger MPS using array notation

 |     |     |     |     |        |
A_1---A_2---A_3---A_4---A_5...---A_N

We can build it as a list of tensors, I guess 
"""

NN = 20

# Build the first element of the MPS A_1 (actually 0th elem of the list)
aMPS = [ np.random.rand(d,chi) + 1j*np.random.rand(d,chi) ]
sMPS = [ [1,2] ]
sMPSconj = [ [1,3] ]

#Build all the others, A_2 to A_(N-1)
for ii in range(2,NN):  # 2 to N-1, so actually building N-2 elements
  aMPS.append(np.random.rand(chi,d,chi) + 1j*np.random.rand(chi,d,chi) )
  sMPS.append(     [2 + 3*(ii-2), 4 + 3*(ii-2), 5 + 3*(ii-2)])
  sMPSconj.append( [3 + 3*(ii-2), 4 + 3*(ii-2), 6 + 3*(ii-2)])
  print(f"Building A{ii}, indices {sMPS[-1]}")
  print(f"and A*{ii}, indices {sMPSconj[-1]}")


# Build the Nth element (actually element N-1 of the list)
aMPS.append( np.random.rand(chi,d) + 1j*np.random.rand(chi,d))
sMPS.append([5 + 3*(NN-3),7 + 3*(NN-3)])
sMPSconj.append([6 + 3*(NN-3),7 + 3*(NN-3)])

#This should be easier 
aMPSconj = [np.conj(elem) for elem in aMPS]

faveOrder = [1 + 3*ii for ii in range(0,NN)]
faveOrder.extend( [2+3*ii for ii in range(0,NN-1)])
faveOrder.extend( [3+3*ii for ii in range(0,NN-1)])

"""
 Try and see if the norm is real 

 There are probably more elegant ways to do it, but first just append mpsconj to mps 
 and structureconj to structure

 and nparray it all, so ncon doesn't complain.. 
"""

aMPS.extend(aMPSconj)
sMPS.extend(sMPSconj)

print(f"order {faveOrder} ")

norm1 = ncon(aMPS, sMPS) 

norm2 = ncon(aMPS, sMPS, faveOrder) 

print(sMPS)
print(f"Norm: contr1={norm1} , contr2={norm2}, diff= {norm1-norm2}")
print(f"normdiff={(norm2-norm1)/np.real(norm1)}, {(norm2-norm1)/np.real(norm2)}")
