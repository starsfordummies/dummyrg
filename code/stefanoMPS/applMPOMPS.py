import myMPSstuff as mps
import myMPOstuff as mpo

from tensornetwork import ncon
import numpy as np 



def applyMPOtoMPS( inMPO: mpo.myMPO, inMPS: mps.myMPS) -> mps.myMPS:

    """ Calculate the product of an MPO with an MPS """

    if inMPO.DD != inMPS.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")

    if inMPO.LL != inMPS.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
    

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
    
    # newMPS = []
    newMPS = [None]*inMPS.LL

    for jj, (Mj, Wj) in enumerate(zip(inMPS.MPS, inMPO.MPO)):  #range(0, inMPS.LL):
        # mps: vL vR p*  | mpo : vL vR pU pD* 
       
        # temp = ncon( [inMPS.MPS[jj], inMPO.MPO[jj] ], [[-1,-3,1],[-2,-4,-5,1]] )
        # newMPS.append(temp.reshape(inMPS.chis[jj]*inMPO.chis[jj], inMPS.chis[jj+1]*inMPO.chis[jj+1], inMPS.DD))
    
        temp = ncon( [Mj, Wj], [[-1,-3,1],[-2,-4,-5,1]] )
        newMPS[jj] = temp.reshape(inMPS.chis[jj]*inMPO.chis[jj], inMPS.chis[jj+1]*inMPO.chis[jj+1], inMPS.DD)

    # Sanity check 
    #if any(n.any() == None for n in newMPS): raise ValueError("didn't build all elems of newMPS")
    
    MPOtimesMPS = mps.myMPS(newMPS)

    return MPOtimesMPS



def expValMPO(psi: mps.myMPS, oper: mpo.myMPO ) -> complex:

        from applMPOMPS import applyMPOtoMPS

        if not psi.canon: psi.bringCan(epsTrunc=1e-12)

        # Apply the MPO to the *ket*, otherwise we might need to conjugate it.. 
        Opsi = applyMPOtoMPS(oper, psi)
        res = mps.voverlap(Opsi, psi, conjugate=True)
        
        return np.real_if_close(res)


def build_environments(psi: mps.myMPS, o: mpo.myMPO) -> list[np.array]:
    # Build the left and right envs for DMRG 

    def build_left_env(psi: mps.myMPS, o: mpo.myMPO, j: int):
        
        if j < 0 or j > psi.LL:
            raise ValueError("left env out of MPS bounds")

        left_env = [np.array(1.).reshape(1,1,1)]*((psi.LL)+1)
        for jj, (Aj, Wj) in enumerate(zip(psi.MPS, o.MPO)):
            # mps: vL vR p*  | mpo : vL vR pU pD* 
            temp = ncon([left_env[jj], Aj], [[-1,-2,1],[1,-3,-4]])
            temp = ncon([temp, Wj],[[-1,2,-3,4],[2,-2,-4,4]])
            left_env[jj+1] = ncon([temp, np.conj(Aj)],[[1,-2,-3,4],[1,-1,4]])

        return left_env

    def build_right_env(psi: mps.myMPS, o: mpo.myMPO, j: int):
        
        if j < 0 or j > psi.LL:
            raise ValueError("right env out of MPS bounds")

        right_env = [np.array(1.).reshape(1,1,1)]*((psi.LL)+1)
        for jj, (Bj, Wj) in enumerate(zip(psi.MPS[::-1], o.MPO[::-1])):
            # mps: vL vR p*  | mpo : vL vR pU pD* 
            temp = ncon([Bj, right_env[psi.LL-jj]], [[-3,1,-4],[-1,-2,1]])
            temp = ncon([Wj, temp],[[-2,2,-4,4],[-1,2,-3,4]])
            right_env[psi.LL-jj-1] = ncon([np.conj(Bj),temp],[[-1,1,4],[1,-2,-3,4]])

        return right_env


    if o.DD != psi.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={o.DD} vs {psi.DD}),  aborting")

    if o.LL != psi.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={o.LL} vs {psi.LL}),  aborting")
    
    if not psi.canon: psi.bringCan(epsTrunc=1e-12)


    le = build_left_env(psi, o, psi.LL)
    re = build_right_env(psi, o, psi.LL)
    #build_right_env(psi, o, j )
    return le, re