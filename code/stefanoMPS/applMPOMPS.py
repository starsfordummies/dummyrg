
import numpy as np

import logging

from myMPSstuff import myMPS
from myMPOstuff import myMPO

from tensornetwork import ncon



def applyMPOtoMPS( inMPO: myMPO, inMPS: myMPS) -> myMPS:

    """ Calculate the product of an MPO with an MPS """

    if inMPO.DD != inMPS.DD: 
        logging.error(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")
        return inMPS

    if inMPO.LL != inMPS.LL: 
        logging.error(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
        return inMPS
    

    """ ncon into new MPS

     We do it site by site,  say we contract with the lower MPO leg 
    TODO: check all conjugation conventions etc 
        
           |d
        D--o--D                |d
           |                   |
         d |     ~    (chi*D)==O==(chi*D)
      chi--O--chi  

        We need some leg reshaping, that's why I write explicitly the leg dimensions
        """
    
    newMPS = []

    for jj in range(0, inMPS.LL):
        # FIXME CHECK NEW CONVENTIONS
        # mps: vL vR p*  | mpo : vL vR pU pD* 
       
        temp = ncon( [inMPS.MPS[jj], inMPO.MPO[jj] ], [[-1,-3,1],[-2,-4,-5,1]] )
        newMPS.append(temp.reshape(inMPS.chis[jj]*inMPO.chis[jj], inMPS.chis[jj+1]*inMPO.chis[jj+1], inMPS.DD))


    MPOMPS = myMPS(newMPS)

    return MPOMPS