
import myMPSstuff as mps
import myMPOstuff as mpo

from tensornetwork import ncon
import numpy as np 


def init_env(LL: int):
    
    env = [np.array(1.).reshape(1,1,1)]*(LL+1)
  
    return env


def build_left_env(psi: mps.myMPS, o: mpo.myMPO):
    
    # FIXME: we don't really need the last element of the L env, so we can truncate at [:-1]

    left_env = init_env(psi.LL)

    for jj, (Aj, Wj) in enumerate(zip(psi.MPS, o.MPO)):
        # mps: vL vR p*  | mpo : vL vR pU pD* 
        #print(f"ncon-ing L[{jj}] with A[{jj}] W[{jj}] A[{jj}]")
        temp = ncon([left_env[jj], Aj], [[-1,-2,1],[1,-3,-4]])
        temp = ncon([temp, Wj],[[-1,2,-3,4],[2,-2,-4,4]])
        left_env[jj+1] = ncon([temp, np.conj(Aj)],[[1,-2,-3,4],[1,-1,4]])

    return left_env



def update_left_env(lenv: list[np.array], Aj: np.array, wj: np.array, jj: int ):
    """ Updates the left environment with the new matrix A[j]
     corresponding to the j-th site.
    So eg. if we feed an updated A3, we will build an updated contraction
      / A3*-       /-
    L3- W3-   =  L4 - 
      \ A3-        \-

    """
    
    temp = ncon([lenv[jj], Aj], [[-1,-2,1],[1,-3,-4]])
    temp = ncon([temp, wj],[[-1,2,-3,4],[2,-2,-4,4]])

    lenv[jj+1] = ncon([temp, np.conj(Aj)],[[1,-2,-3,4],[1,-1,4]])

    return lenv # though we already updated it in place 




def build_right_env(psi: mps.myMPS, o: mpo.myMPO):
    
    # FIXME: we don't really need the 1st element of the R env, so we can truncate at [1:]

    right_env = init_env(psi.LL)

    for jj, (Bj, Wj) in enumerate(zip(psi.MPS[::-1], o.MPO[::-1])):
        # mps: vL vR p*  | mpo : vL vR pU pD* 
        rjj = -jj-1 

        temp = ncon([Bj, right_env[rjj]], [[-3,1,-4],[-1,-2,1]])
        temp = ncon([Wj, temp],[[-2,2,-4,4],[-1,2,-3,4]])
        right_env[rjj-1] = ncon([np.conj(Bj),temp],[[-1,1,4],[1,-2,-3,4]])
    return right_env






def update_right_env(renv: list[np.array], Bj: np.array, wj: np.array, jj: int ):
    """ Updates the right environment with the new matrix B[j] 
    corresponding to the j-th site. 
    So eg. if we feed an updated B3, we will build an updated contraction
    - B2*-\        \
    - W2-- R3  =   -R2 
    - B2--/        / 
    """
    

    temp = ncon([Bj, renv[jj+1]], [[-3,1,-4],[-1,-2,1]])
    temp = ncon([wj, temp],[[-2,2,-4,4],[-1,2,-3,4]])
    renv[jj] = ncon([np.conj(Bj),temp],[[-1,1,4],[1,-2,-3,4]])

    return renv


def build_environments(psi: mps.myMPS, o: mpo.myMPO) -> tuple[list[np.array],list[np.array]]:
    # Build the left and right envs for DMRG 

    if o.DD != psi.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={o.DD} vs {psi.DD}),  aborting")

    if o.LL != psi.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={o.LL} vs {psi.LL}),  aborting")
    
    if not psi.canon: psi.bringCan(epsTrunc=1e-12)


    le = build_left_env(psi, o)
    re = build_right_env(psi, o)

    return le, re