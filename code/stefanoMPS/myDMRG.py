import myMPSstuff as mps 
import myMPOstuff as mpo
import myEnvironmentsMPO as envs

import numpy as np 
from tensornetwork import ncon
from scipy import linalg as LA 
import scipy.sparse.linalg as LAS


def findGS_DMRG( inMPO : mpo.myMPO, inMPS: mps.myMPS) -> mps.myMPS:

    if inMPO.DD != inMPS.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")

    if inMPO.LL != inMPS.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
    
    LL = inMPS.LL 

    if not inMPS.canon: inMPS.bringCan(epsTrunc=1e-12)

    # build environments
    #le, re = envs.build_environments(inMPS, inMPO)

    le = envs.init_env(inMPS.LL)
    re = envs.build_right_env(inMPS, inMPO)

    # Build Heff 
    chis = inMPS.chis
    ww = inMPO.MPO
    Apsi = inMPS.MPS
    dd = inMPO.DD

    nsweeps = 4

    toleig = 1e-6


    for ns in range(0,nsweeps):

        toleig = toleig*0.5
      
        # ===>>===>     (L-R sweep )
        for jj in range(0,LL-1): #, Aj in enumerate(Apsi[:-1]):

            Heff = ncon([le[jj], ww[jj], ww[jj+1], re[jj+2]],
            [[-1,1,-5],[1,2,-2,-6],[2,3,-3,-7],[-4,3,-8]])

            Heff = Heff.reshape( chis[jj]*dd*dd*chis[jj+2], chis[jj]*dd*dd*chis[jj+2])

            lam0, eivec0 = LAS.eigsh(Heff, k=1, which='SA',tol=toleig) #, v0=psi_flat, tol=tol, ncv=N_min)
            #print(f"{lam0 = }")
            u, s, vdag, chiTrunc = mps.SVD_trunc(eivec0.reshape(chis[jj]*dd,dd*chis[jj+2]),1e-14,40)
            print(f"[L] truncating {chis[jj]*dd} {dd*chis[jj+2]} -> {chiTrunc}")
            ss = LA.sqrtm(np.diag(s))

            uss = (u @ ss).reshape(chis[jj],chiTrunc,dd)
            ssv = (ss @ vdag).reshape(chiTrunc,chis[jj+2],dd) 

            Apsi[jj] = uss
            Apsi[jj+1] = ssv

            chis[jj+1] = chiTrunc

            #print(f"truncated {np.shape(Apsi[jj])}, {np.shape(Apsi[jj+1])}")


            # rebuild left env #TODO: do update instead of full rebuild
            le = envs.update_left_env(le, Apsi[jj], ww[jj], jj)
 
        print( [np.shape(i) for i in le])


        # <====<<====    (R-L sweep)
        Emin = 0.
        for jj in range(LL,1,-1): # in enumerate(Apsi[1:]):

            Heff = ncon([le[jj-2], ww[jj-2], ww[jj-1], re[jj]],
            [[-1,1,-5],[1,2,-2,-6],[2,3,-3,-7],[-4,3,-8]])
            
            Heff = Heff.reshape( chis[jj-2]*dd*dd*chis[jj], chis[jj-2]*dd*dd*chis[jj])
            lam0, eivec0 = LAS.eigsh(Heff, k=1, which='SA',tol=toleig) #, v0=psi_flat, tol=tol, ncv=N_min)
            u, s, vdag, chiTrunc = mps.SVD_trunc(eivec0.reshape(chis[jj-2]*dd,dd*chis[jj]),1e-14,40)
            print(f"[R] truncating {chis[jj-2]*dd} {dd*chis[jj]} -> {chiTrunc}")
            ss = LA.sqrtm(np.diag(s))

            uss = (u @ ss).reshape(chis[jj-2],chiTrunc,dd)
            ssv = (ss @ vdag).reshape(chiTrunc,chis[jj],dd) 


            Apsi[jj-2] = uss
            Apsi[jj-1] = ssv
            
            chis[jj-1] = chiTrunc


            # rebuild right env #TODO: do update instead of full rebuild
            #re = envs.build_right_env(inMPS,inMPO)
            re = envs.update_right_env(re, Apsi[jj-1], ww[jj-1], jj-1)

            if lam0 < Emin : Emin = lam0

        print(f"Nsweep = {ns}, En = {Emin}")
        print(f"chis = {inMPS.chis}")

    return Emin 