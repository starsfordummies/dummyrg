from myMPSstuff import myMPS
from myMPOstuff import myMPO

from tensornetwork import ncon


def findGS_DMRG( inMPO : myMPO, inMPS: myMPS) -> myMPS:

    if inMPO.DD != inMPS.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")

    if inMPO.LL != inMPS.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
    

    if(inMPS.form != 'R'): inMPS.bringCan(mode='C',epsTrunc=1e-12)
