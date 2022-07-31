# Playing around with shapes and reshapes 


import numpy as np
import scipy.linalg as LA

from ncon import ncon



sx = np.asarray([[0., 1], [1, 0.]])
sy = np.asarray([[0., -1j], [+1j, 0.]])
sz = np.asarray([[1, 0.], [0., -1]])


test = np.asarray([[ 0.1*sx, 0.2*sy], [sz, 0.3*sz ]])

print(type(test))
test[0][1]

#so it's [virt][virt][phys][phys]

np.shape(test)
#np.reshape(test,(2,2,2,2))

# I guess this gives M_ijlk M*_i'j'k'l'  (indices in this order)
test2 = np.tensordot(test, np.conj(test), axes = 0)
# So to get it to M2_ii'jj'kk'll' I should do something like
test2t = np.transpose(test2, [0,4,1,5,2,6,3,7])

# Can I get the same with ncon? Naively test2 should be the same as
test2nc = ncon([test,np.conj(test)],[[-1,-2,-3,-4],[-5,-6,-7,-8]])
# Could I reshape it directly here by doing 
test2nct = ncon([test,np.conj(test)],[[-1,-3,-5,-7],[-2,-4,-6,-8]])

print(np.allclose(test2, test2t))
print(np.allclose(test2, test2nc))
print(np.allclose(test2t, test2nct))

print(np.shape(test2t))

# Now we can reshape 

np.reshape(test2t,(4,4,4,4));
