import numpy as np
from tensornetwork import ncon
import logging

from myMPSstuff import myMPS
from myMPOstuff import myMPO

def applyMPOtoMPS( inMPO: myMPO, inMPS: myMPS, trunc: float = 1e-10, chiMax: int = 40, canon: bool = True):

    """ Calculate the product of an MPO with an MPS """

    if inMPO.DD != inMPS.DD: 
        logging.error(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")
        return inMPS

    if inMPO.LL != inMPS.LL: 
        logging.error(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
        return inMPS
    

    # FIXME: do we need to put in can form before applying ?

    #if inMPS.form != 'R':
    #    logging.warning("Bringing input MPS to right canonical form")
    #    inMPS.bringCan('LR', 1e-12)
    #    print(f"norm before = {inMPS.getNorm}")

    #print(inMPS.getIndices())
    #print(inMPO.getIndices())

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
        # Recall mps: vL p* vR  | mpo : vL vR pU pD* 
        temp = ncon( [inMPS.MPS[jj], inMPO.MPO[jj] ], [[-1,2,-3],[-2,-4,-5,2]] )
        newMPS.append(temp.reshape(inMPS.chis[jj]*inMPO.chis[jj] , inMPS.DD, inMPS.chis[jj+1]*inMPO.chis[jj+1]))


    MPOMPS = myMPS(newMPS)
  
    if canon:
        MPOMPS.bringCan('R', epsTrunc=trunc, chiMax = chiMax)
    #print(MPOMPS.chis)

    #print(f"norm after can = {MPOMPS.getNorm()}")
    #print(f"is it (al least approx) normalized? = {MPOMPS.normalized}")


    return MPOMPS