import numpy as np
from tensornetwork import ncon

mats = [ np.random.rand(1,4) + 1j*np.random.rand(1,4) ]
mats.append( np.random.rand(4,5) + 1j*np.random.rand(4,5) )
mats.append( np.random.rand(5,7) + 1j*np.random.rand(5,7) )
mats.append( np.random.rand(7,1) + 1j*np.random.rand(7,1) )

idss = [ [-1,1],[1,2],[2,3],[3,-2] ]

n1 = ncon(mats,idss)

# :)

n1c = n1.conj()

mc = [m.conj() for m in mats]

n1cAlt = ncon(mc,idss)

mats2 = mats[::-1]
mats2 = [m.conj().transpose() for m in mats2]

