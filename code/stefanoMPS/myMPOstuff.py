# Last modified: 2022/07/12 16:20:28

import numpy as np
from numpy import linalg as LA
from tensornetwork import ncon
import logging

#import math 
#import copy 

# v_L , v_R , phys_U, phys_D 


def randMPO(LL: int, chi: int=5, d: int=2):

    """ MPO Builder function - OBC for now 
    Returns a list of length LL of random tensors with bond dimension chi 
    and physical dimension d 

                                                          3  
                                                          |
    I will use here Luca's index convention:          1 - O - 2
            v_L , v_R , phys_U, phys_D                    |
                                                          4
    

    """

    logging.info(f"Building random MPO with length {LL}, chi = {chi} and physical d={d}")

    # Build the first element of the MPO W_1 (actually 0th elem of the list)
    myMPO = [ np.random.rand(1,chi,d,d) + 1j*np.random.rand(1,chi,d,d) ]

    #Build all the others, W_2 to W_(N-1)
    for _ in range(2,LL):  # 2 to N-1, so actually building N-2 elements
         myMPO.append(np.random.rand(chi,chi,d,d) + 1j*np.random.rand(chi,chi,d,d) )
 
    # Build the Nth element (actually element N-1 of the list)
    myMPO.append( np.random.rand(chi,1,d,d) + 1j*np.random.rand(chi,1,d,d))

    return myMPO


def isingMPO(LL: int, J: float = 1., g: float = 0.5):
    """ Ising Hamiltonian in MPO form """
    sx = np.array([[0.,1],[1,0]])
    sz = np.array([[1.,0],[0,-1]])
    id = np.eye(2)

    Wmpo = [id]*LL
    
    Wmpo[0] = np.array([[g*sx , J*sz, id]])
    for j in range(1,LL-1):
        Wmpo[j] = np.array([[id, 0.*id, 0.*id],[sz, 0.*id, 0.*id], [g*sx, J*sz, id]])
    Wmpo[LL-1] = np.array([[id],[J*sz],[id]])

    print(f"Ising MPO, parameters: J={J} g={g},  shapes:")
    print([np.shape(w) for w in Wmpo])
    
    return Wmpo

    

class myMPO:

    """ My own MPO implementation - Contains the following elements:
    LL(length),
    DD (physical dim), 
    chis(bond dims), 
    indices (for contraction),

    offIndices is the offset factor which we use to label the virtual indices to be contracted, 
    we take it to be >> 1 so we don't get mixed indices labels 
    """

    def __init__(self, inputMPO: list=randMPO(7), offIndices: int=95):
      
        LL = len(inputMPO)

        DD = np.shape(inputMPO[1])[3]  # not the most elegant way to extract it but eh..

        self.MPO = inputMPO 


        mChi = [ np.shape(mm)[0] for mm in inputMPO ]
        mChi.append(np.shape(inputMPO[-1])[1])
        
        #inputMPS[1:] ]  # Should be LL-1 long
      

        print(f"MPO with length {LL} and physical d={DD}")
        print(f"chi {mChi}")


        """
        # Should we use a DICT for the indices? 
        # For a given site jj, 
        indices = [{ 
            'vL': jj + offIndices*LL,
            'vR': jj + offIndices*LL+1
            'phU': jj,
            'phD': jj + LL,
            }]

        """

        indices=[]
        for midx in range(1,LL+1):  #Builds LL sets of indices from 1 to LL
            indices.append({ 'vL': offIndices*LL+midx, 'vR': offIndices*LL+midx+1, 'phU': midx, 'phD': midx+ LL })

        self.LL = LL  
        self.DD = DD  

        self.indices = indices
  
        self.chis = mChi
   

     
     
    def getIndices(self):
        """ Returns a list of lists with the indices for contracting the TN with ncon """
        iL = []
        for el in self.indices:

            # Build the indices list
            idxs = []
            
            if el["vL"] != 0:
                idxs.append(el["vL"])

            idxs.append(el["phU"])
            idxs.append(el["phD"])

            if el["vR"] != 0:
                idxs.append(el["vR"])
            
            iL.append(idxs)
        
        return iL



    def getPhysIndices( self ):
        """ Returns a list with the labels of the physical indices """
       
        indices = self.indices
        listPhys = []
        [listPhys.append([idx["phU"], idx["phD"]]) for idx in indices]

        return listPhys




    def updatePhysIndices( self, openIndices : list ):
        """ Updates the labels of the physical indices using an input list """
        if len(openIndices) != (self.LL):
            logging.warning("Wrong length of open indices list, doing nothing")
            pass
        else:
            indices = self.indices
            for (ii, idx) in enumerate(indices):
                if(len(idx) != 2):
                    logging.warning("Wrong length of open indices list, doing nothing")
                    pass
                else:
                    idx["phU"] = openIndices[ii]
                    idx["phD"] = openIndices[ii]
            #self.indices = indices  # Is this even necessary ? 


