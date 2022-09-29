# Last modified: 2022/09/14 17:27:43

from math import sin, cos, sinh, cosh, sqrt

import numpy as np
from scipy import linalg as LA
from tensornetwork import ncon
import logging

sx = np.array([[0.,1.],[1.,0.]])
sz = np.array([[1.,0.],[0.,-1.]])
id = np.eye(2)
zero = np.zeros([2,2])

def IsingMPO(LL: int, J: float = 1., g: float = 0.4) -> list[np.array]:

    """ Ising Hamiltonian in MPO form """

    Wmpo = [id]*LL
    
    # I put a minus in front of this to get the correct Ising Ham with an overall minus,
    # alternatively one could just define the couplings as negative I guess
    #         |
    #         v
    
    Wmpo[0] = - np.array([[g*sx, J*sz, id]])

    for j in range(1,LL-1):
        Wmpo[j] = np.array([[id,  zero,   zero],
                            [sz,  zero,   zero], 
                            [g*sx, J*sz,   id ]])
                            
    Wmpo[LL-1] = np.array([[id],[sz],[g*sx]]) 


   # nel mio MPO parto dalla dx e vado verso sx, vs. Luca che va da sx a dx 
   #
   # |0><0| x 1 + 10 x Z -h 20 x X + h 


    logging.info(f"Ising MPO, parameters: J={J} g={g},  shapes:")
    logging.info([np.shape(w) for w in Wmpo])

    print(Wmpo[0])
    
    return Wmpo


def IsingMPO_swapLR(LL: int, J: float = 1., g: float = 0.4) -> list[np.array]:

    """ Ising Hamiltonian in MPO form """

    Wmpo = [id]*LL
    
    # I put a minus in front of this to get the correct Ising Ham with an overall minus,
    # alternatively one could just define the couplings as negative I guess
    #         |
    #         v
    
    Wmpo[0] = - np.array([[g*sx, J*sz, id]])

    for j in range(1,LL-1):
        Wmpo[j] = np.array([[id,  zero,   zero],
                            [sz,  zero,   zero], 
                            [g*sx, J*sz,   id ]])
                            
    Wmpo[LL-1] = np.array([[id],[sz],[g*sx]]) 


   # nel mio MPO parto dalla dx e vado verso sx, vs. Luca che va da sx a dx 
   #
   # |0><0| x 1 + 10 x Z -h 20 x X + h 


    logging.info(f"Ising MPO, parameters: J={J} g={g},  shapes:")
    logging.info([np.shape(w) for w in Wmpo])

    print(Wmpo[0])

    for i,wi in enumerate(Wmpo):
        Wmpo[i] = wi.transpose(0,1,3,2)
    
    return Wmpo





def OneMinusEpsHIsingMPO(LL: int, J: float = 1., g: float = 0.4, eps: float = 0.1) -> list[np.array]:
    """ 1 - eps*H in MPO form, should be seen as a first approx to exp(-eps*H) """

    Wmpo = [id]*LL
    
    Wmpo[0] = np.array([[id + eps*g*sx, +eps*J*sz, +eps*id]])

    for j in range(1,LL-1):
        Wmpo[j] = np.array([[id,  zero,   zero],
                            [sz,  zero,   zero], 
                            [g*sx, J*sz,   id ]])
                            
    Wmpo[LL-1] = np.array([[id],[sz],[g*sx]]) 


   # nel mio MPO parto dalla dx e vado verso sx, vs. Luca che va da sx a dx 
   #
   # |0><0| x 1 + 10 x Z -h 20 x X + h 


    logging.info(f"Ising MPO, parameters: J={J} g={g},  shapes:")
    logging.info([np.shape(w) for w in Wmpo])
    
    return Wmpo
    


def expMinusEpsHIsingMPO(LL: int, J: float = 1., g: float = 0.4, eps: float = 0.1, mode="svd") -> list[np.array]:
    
    """ Exp(-tau*Hising) in MPO form, 
    using second-order Trotter, build with SVD """

    if mode == "svd":
        Ut =np.reshape(LA.expm(eps*np.kron(sz,sz)),(2,2,2,2))
    
        u, s, v = LA.svd( np.reshape(np.transpose(Ut,(0,2,1,3)),(4,4)) )
        # only 2 of the 4 SVs are nonzero, so we can truncate 
        #vss = LA.sqrtm(np.diag(s)) @ v;
        vss = LA.sqrtm(np.diag(s[:2])) @ v[:2,:]
        #ssu = u @ LA.sqrtm(np.diag(s));
        ssu = u[:,:2] @ LA.sqrtm(np.diag(s[:2]))

        MPO = ncon([np.reshape(ssu,(2,2,2)),np.reshape(vss,(2,2,2))],[[-3,1,-2],[-1,1,-4]]) 
        WW = ncon([LA.expm(eps*g*0.5*sx), MPO, LA.expm(eps*g*0.5*sx)],[[-3,1],[-1,-2,1,2],[2,-4]])
        
    elif mode == "sin":
    
        print("Using sinh/cosh decomposition (symmetric form)")
        m11= cosh(eps)*LA.expm(g*sz*eps)
        #m11= cosh(eps)*LA.expm(-g*sz*eps)

        #m12=1j*sqrt(sinh(eps)*cosh(eps))*LA.expm(-g*sz*eps/2.)*sx*LA.expm(-g*sz*eps/2.)
        m12=1j*sqrt(sinh(eps)*cosh(eps))*LA.expm(g*sz*eps/2.)*sx*LA.expm(g*sz*eps/2.)
        m21= m12 
        #m22=-sinh(eps)*LA.expm(-g*sz*eps)
        m22=-sinh(eps)*LA.expm(g*sz*eps)
        
        WW = np.asarray([[ m11, m12],[m21, m22]])
    

    # Fill the MPO matrices
    Wmpo = [WW]* LL
    
    #For the edges 
    # First column: WW[0:2,0,:,:]
    # First row: WW[0,0:2,:,:]
    
    # Wmpo[0] = WW[0:2,0,:,:].reshape(1,2,2,2)
    # Wmpo[LL-1] = WW[0,0:2,:,:].reshape(2,1,2,2)
    
    Wmpo[0] = WW[0,0:2,:,:].reshape(1,2,2,2)
    Wmpo[LL-1] = WW[0:2,0,:,:].reshape(2,1,2,2)
    return Wmpo
    