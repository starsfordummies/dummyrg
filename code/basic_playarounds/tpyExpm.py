import numpy as np
import scipy 
import tenpy.linalg.np_conserved as npc

mma = npc.Array.from_ndarray_trivial(np.arange(4.).reshape(2, 2))
mm = mma.to_ndarray()

print(mma)

print(mm)

print(npc.expm(mma))

print(scipy.linalg.expm(mm))

