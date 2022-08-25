# Last modified: 2022/08/25 16:00:08

import numpy as np
import logging




def randMPO(LL: int, chi: int=5, d: int=2) -> list:

    """ MPO Builder function - OBC for now 
    Returns a list of length LL of random tensors with bond dimension chi 
    and physical dimension d 

                                                          3  
                                                          |
    I will use here Luca's index convention:          1 - W - 2
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

    



class myMPO:

    """ My own MPO implementation - Contains the following elements:
    LL(length),
    DD (physical dim), 
    chis(bond dims), 
    MPO (the matrices)
    """
    
    __slots__ = ['LL','DD','MPO','chis']
    
    def __init__(self, inputWs : list[np.ndarray] = randMPO(7)):
      
        LL = len(inputWs)

        idx = {
             'vL' : 0,
             'vR' : 1,
             'pU' : 3,
             'pD' : 4,
        }
       
        # Physical dimension 
        DD = np.shape(inputWs[1])[idx['pU']]  # not the most elegant way to extract it but eh..
        
        self.MPO = inputWs 

        mChi = [ np.shape(mm)[idx['vL']] for mm in inputWs ]
        mChi.append(np.shape(inputWs[-1])[idx['vR']])
        

        logging.info(f"MPO with length {LL} and physical d={DD}")
        logging.info(f"chi {mChi}")




        self.LL = LL  
        self.DD = DD  
        self.chis = mChi
