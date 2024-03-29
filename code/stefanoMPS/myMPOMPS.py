
import myMPSstuff as mps
import myMPOstuff as mpo

from myUtils import sncon as ncon
from myUtils import real_close as rc
import numpy as np 



def applyMPOtoMPS(inMPO: mpo.myMPO, inMPS: mps.myMPS) -> mps.myMPS:

    """ Calculate the product of an MPO with an MPS """

    if inMPO.DD != inMPS.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")

    if inMPO.LL != inMPS.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
    

    """ ncon into new MPS

     We do it site by site,  say we contract with the lower MPO leg 

        
           |d
        D--o--D                |d
           |                   |
         d |     ~    (chi*D)==O==(chi*D)
      chi--O--chi  

        We need some leg reshaping, that's why I write explicitly the leg dimensions
        """
    
    # newMPS = []
    newMPS = [np.array(None)]*inMPS.LL

    for jj, (Mj, Wj) in enumerate(zip(inMPS.MPS, inMPO.MPO)):  #range(0, inMPS.LL):
        # mps: vL vR p*  | mpo : vL vR pU pD* 
       
        # temp = ncon( [inMPS.MPS[jj], inMPO.MPO[jj] ], [[-1,-3,1],[-2,-4,-5,1]] )
        # newMPS.append(temp.reshape(inMPS.chis[jj]*inMPO.chis[jj], inMPS.chis[jj+1]*inMPO.chis[jj+1], inMPS.DD))
    
        temp = ncon( [Mj, Wj], [[-1,-3,1],[-2,-4,-5,1]] )
        newMPS[jj] = temp.reshape(inMPS.chis[jj]*inMPO.chis[jj], inMPS.chis[jj+1]*inMPO.chis[jj+1], inMPS.DD)

    MPOtimesMPS = mps.myMPS(newMPS)

    return MPOtimesMPS



def expValMPO(psi: mps.myMPS, oper: mpo.myMPO ) -> complex:

        psi.checkNormalized()
        if not psi.canon: psi.bringCan(epsTrunc=1e-12)
        if not psi.normalized: psi.bringCan(epsTrunc=1e-12)

        # Apply the MPO to the *ket*, otherwise we might need to conjugate it.. 
        Opsi = applyMPOtoMPS(oper, psi)
        res = mps.voverlap(Opsi, psi, conjugate=True)
        
        return rc(res)





