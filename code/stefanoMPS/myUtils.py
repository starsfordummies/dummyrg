import numpy as np 
from ncon import ncon 

def checkIdMatrix(ainp: np.ndarray, epstol = 1e-14) -> bool:
    """Checks if an array is an identity matrix (within machine precision)"""

    a = np.array(ainp)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Not even square: {a.shape}")
    else:
        size = a.shape[0]
        if np.all(np.abs(a - np.eye(size)) < epstol):
            print(f"identity, size = {size}")
            return True
        else:
            print(f"Square but not id, difference Max = {np.max(np.abs(a - np.eye(size)))}")
            return False


# TODO:
def myReshape(idx: dict, tensor: np.ndarray):
    pass

def alleq(tensor1: np.ndarray, tensor2:np.ndarray) -> np.bool_:
  if np.shape(tensor1) != np.shape(tensor2): return np.False_

  return np.ravel(np.isclose(tensor1,tensor2)).all()



def build_perms(tensor: np.ndarray, n: int, perm: list, listPerms: list) -> list:
    
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

          build_perms(tensor, n-1, tperm, outList)

          #print(f"output: {outList}")

    return outList
          

      #perm(tensor,n-1)
    


def equal_up_to_perm(t1: np.ndarray, t2: np.ndarray) -> bool:
    # First check if shapes are different
    s1 = np.shape(t1)
    s2 = np.shape(t2)
    print(s1,s2)
    if sum(s1) != sum(s2):
        print("Cannot possibly reshape one into another, quitting")
        return False
    else:
        print(f"Shapes = {s1}, {s2}, checking..")

    # if shapes are different, it should give a hint on how to reshape..

    if alleq(t1,t2):
        print("Arrays are equal")
        return True 
    else: 
        return False



""" Example usage: 

  c0 = [i for i in range(0,len(s2))]
  lps = []
  build_perms(t2, len(s2), c0, lps)

  print(lps)

  for shuf in lps:
    if alleq(t1,t2.transpose(shuf)):
        print(f"Equal after reshuffling {shuf}")
        return True

"""


def sncon(listArr, listInd) -> np.ndarray:
    try:
        return ncon(listArr,listInd)
    except ValueError:
        print(f"wrong contraction")
        print(f"shapes: [{[np.shape(a) for a in listArr]}]")
        print(f"contrs: {listInd}")
        return np.array(np.nan)


def real_close(input: np.ndarray | np.complex128) -> np.ndarray | np.complex128 | np.float64: 
    return np.real_if_close(input)
