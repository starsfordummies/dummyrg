
import myMPSstuff as mps
import myMPOstuff as mpo

from tensornetwork import ncon
import numpy as np 


def build_left_env(psi: mps.myMPS, o: mpo.myMPO, j: int = -99 ):
    
    # FIXME: ugly hack for later when we want to update 
    if j == -99: j = psi.LL

    if j < 0 or j > psi.LL:
        raise ValueError("left env out of MPS bounds")

    left_env = [np.array(1.).reshape(1,1,1)]*((psi.LL)+1)
    for jj, (Aj, Wj) in enumerate(zip(psi.MPS, o.MPO)):
        # mps: vL vR p*  | mpo : vL vR pU pD* 
        #print(f"ncon-ing L[{jj}] with A[{jj}] W[{jj}] A[{jj}]")
        temp = ncon([left_env[jj], Aj], [[-1,-2,1],[1,-3,-4]])
        temp = ncon([temp, Wj],[[-1,2,-3,4],[2,-2,-4,4]])
        left_env[jj+1] = ncon([temp, np.conj(Aj)],[[1,-2,-3,4],[1,-1,4]])

    return left_env




def build_right_env(psi: mps.myMPS, o: mpo.myMPO, j: int = -99):
    
    if j == -99: j = psi.LL

    if j < 0 or j > psi.LL:
        raise ValueError("right env out of MPS bounds")

    right_env = [np.array(1.).reshape(1,1,1)]*((psi.LL)+1)
    for jj, (Bj, Wj) in enumerate(zip(psi.MPS[::-1], o.MPO[::-1])):
        # mps: vL vR p*  | mpo : vL vR pU pD* 
        rjj = -jj-1
        temp = ncon([Bj, right_env[rjj]], [[-3,1,-4],[-1,-2,1]])
        temp = ncon([Wj, temp],[[-2,2,-4,4],[-1,2,-3,4]])
        right_env[rjj-1] = ncon([np.conj(Bj),temp],[[-1,1,4],[1,-2,-3,4]])
        print(rjj-1, np.shape(Bj), np.shape(Wj), np.shape(right_env[rjj]), np.shape(right_env[rjj-1]))
    return right_env


def build_environments(psi: mps.myMPS, o: mpo.myMPO) -> list[np.array]:
    # Build the left and right envs for DMRG 

    if o.DD != psi.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={o.DD} vs {psi.DD}),  aborting")

    if o.LL != psi.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={o.LL} vs {psi.LL}),  aborting")
    
    if not psi.canon: psi.bringCan(epsTrunc=1e-12)


    le = build_left_env(psi, o, psi.LL)
    re = build_right_env(psi, o, psi.LL)
    #build_right_env(psi, o, j )
    return le, re