import numpy as np 
from tensornetwork import ncon

a = np.ones((2,3,4,5,6,7,8))
b = np.ones((3,4,5,3,4,6))

def mynconwrap(listArr, listInd):
    try:
        return ncon(listArr,listInd)
    except ValueError:
        print(f"wrong contraction")
        print(f"shapes: [{np.shape(a)}], [{np.shape(b)}]")
        print(f"contrs: {listInd}")



c = mynconwrap([a,b],[[-1,1,2,-3,4,-5,-6],[1,2,-2,-7,-8,4]])
print(c)
c = mynconwrap([a,b],[[1,-1,2],[1,2,-2]])
