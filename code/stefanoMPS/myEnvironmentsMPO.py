
import myMPSstuff as mps
import myMPOstuff as mpo

from myUtils import sncon as ncon
import numpy as np 


def init_env(LL: int) -> list[np.ndarray]:
    """ Initializes left or right environment array of length L+1, filled with 1 """
    
    env = [np.array(1.).reshape(1,1,1)]*(LL+1)
    return env


# TODO: option for building environments with "working" convention vL dd vR 

def build_left_env(psi: mps.myMPS, o: mpo.myMPO, workConv = False):
    
    # FIXME: we don't really need the last element of the L env, so we can truncate at [:-1]

    left_env = init_env(psi.LL)

    for jj, (Aj, Wj) in enumerate(zip(psi.MPS[:-1], o.MPO[:-1])):
        # mps: vL vR p*  | mpo : vL vR pU pD* 

        if workConv:
            # working convention: MPS tensor indices are (vL, ph, vR)
            left_env[jj+1] = ncon([left_env[jj], Aj, Wj, np.conj(Aj)],
                                [[],[],[3,-2,1,5],[2,1,-1]])
        else:
            left_env[jj+1] = ncon( [left_env[jj], Aj, Wj, np.conj(Aj)],
                       [[4,2,1],[1,-3,3],[2,-2,5,3],[4,-1,5]])
    return left_env



def update_left_env(lenv: list[np.ndarray], Aj: np.ndarray, wj: np.ndarray, jj: int ):
    """ Updates lenv[jj+1] using lenv[j],A[j],W[j] 
     corresponding to the j-th site.
    So eg. if we feed an updated A3, we will build an updated contraction
      / A3*-       /-
    L3- W3-   =  L4 - 
      \\ A3-       \\-

    """
    
    lenv[jj+1] = ncon( [lenv[jj], Aj, wj, np.conj(Aj)],
                       [[4,2,1],[1,-3,3],[2,-2,5,3],[4,-1,5]])
    #print(f"updating L[{jj+1}]")

    return 0 




def build_right_env(psi: mps.myMPS, o: mpo.myMPO):
    
    # FIXME: we don't really need the 1st element of the R env, so we can truncate at [1:]

    right_env = init_env(psi.LL)

    for jj, (Bj, Wj) in enumerate(zip(psi.MPS[:0:-1], o.MPO[:1:-1])):
        # mps: vL vR p*  | mpo : vL vR pU pD* 
        rjj = -jj-1 

        right_env[rjj-1] = ncon([right_env[rjj], Bj, Wj, np.conj(Bj)],
                            [[4,2,1],[-3,1,3],[-2,2,5,3],[-1,4,5]])
        #print(f"updating R[{rjj-1}]")
    return right_env






def update_right_env(renv: list[np.ndarray], Bj: np.ndarray, wj: np.ndarray, jj: int ):
    r""" Builds renv[j] by contracting R(j+1) with the input B_{j+1}-W_{j+1} 
    So eg. if we feed an updated B3, we will build an updated contraction
    - B2*-\\       \
    - W2-- R3  =   -R2 
    - B2--/        / 
    """
    

    renv[jj] = ncon([renv[jj+1], Bj, wj, np.conj(Bj)],
                            [[4,2,1],[-3,1,3],[-2,2,5,3],[-1,4,5]])

    return 0


def build_environments(psi: mps.myMPS, o: mpo.myMPO) -> tuple[list[np.ndarray],list[np.ndarray]]:
    """ Builds the left and right envs for psi and O """

    if o.DD != psi.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={o.DD} vs {psi.DD}),  aborting")

    if o.LL != psi.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={o.LL} vs {psi.LL}),  aborting")
    
    if not psi.canon: psi.bringCan(epsTrunc=1e-12)

    le = build_left_env(psi, o)
    re = build_right_env(psi, o)

    return le, re