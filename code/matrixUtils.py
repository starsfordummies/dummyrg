import numpy as np 

def checkIdMatrix(ainp: np.array, epstol = 1e-14):
    """Checks if an array is an identity matrix (within machine precision)"""

    a = np.array(ainp)
    if a.shape[0] != a.shape[1]:
        print(f"Not even square: {a.shape}")
        return False
    else:
        size = a.shape[0]
        if np.all(np.abs(a - np.eye(size)) < epstol):
            print(f"identity, size = {size}")
            return True
        else:
            print(f"Square but not id, difference = {np.abs(a - np.eye(size))}")
            return False