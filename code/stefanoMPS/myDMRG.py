import myMPSstuff as mps 
import myMPOstuff as mpo
import applMPOMPS as mpomps

import numpy as np 
from tensornetwork import ncon
from scipy import linalg as LA 
import scipy.sparse.linalg as LAS


def findGS_DMRG( inMPO : mpo.myMPO, inMPS: mps.myMPS) -> mps.myMPS:

    if inMPO.DD != inMPS.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")

    if inMPO.LL != inMPS.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
    

    if not inMPS.canon: inMPS.bringCan(epsTrunc=1e-12)

    # build environments
    le, re = mpomps.build_environments(inMPS, inMPO)

    # Build Heff 
    chis = inMPS.chis
    ww = inMPO.MPO
    Apsi = inMPS.MPS
    dd = inMPO.DD

    nsweeps = 5

    for ns in range(0,nsweeps):
        # L->R sweep 
        for jj, Aj in enumerate(Apsi[:-1]):
            print(np.shape(le[jj]), np.shape(ww[jj]), np.shape(ww[jj+1]), np.shape(re[jj+2]))
            print(f"chis = {chis[jj]}, {chis[jj+2]}")
            print(f"working with chis {jj} and {jj+2}, we will change chi {jj+1}")

            Heff = ncon([le[jj], ww[jj], ww[jj+1], re[jj+2]],
            [[-1,1,-5],[1,2,-2,-6],[2,3,-3,-7],[-4,3,-8]])

            Heff = Heff.reshape( chis[jj]*dd*dd*chis[jj+2], chis[jj]*dd*dd*chis[jj+2])

            lam0, eiv0 = LAS.eigsh(Heff, k=1, which='SA',tol=1e-6) #, v0=psi_flat, tol=tol, ncv=N_min)

            u, s, vdag, chiTrunc = mps.SVD_trunc(eiv0.reshape(chis[jj]*dd,dd*chis[jj+2]),1e-12,40)

            ss = LA.sqrtm(np.diag(s))

            uss = (u @ ss).reshape(chis[jj],chiTrunc,dd)
            ssv = (ss @ vdag).reshape(chiTrunc,chis[jj+2],dd) 

            print(f"chitrunc = {chiTrunc}")
            print(f"Replacing A[{jj}]{np.shape(Apsi[jj])} with {np.shape(uss)} ")
            print(f"Replacing A[{jj+1}]{np.shape(Apsi[jj+1])} with {np.shape(ssv)} ")

            Apsi[jj] = uss
            Apsi[jj+1] = ssv

            chis[jj+1] = chiTrunc

            # rebuild left env #TODO: do update instead of full rebuild
            print(f"updating left env, {jj} elem from ..")
            le = mpomps.build_left_env(inMPS,inMPO)


    # R->L sweep 
        Emin = 0.
        #for jj, Bj in enumerate(Apsi[1::-1]):
        for jj, Bj in enumerate(Apsi):
            rjj = -jj

            print(np.shape(le[rjj-2]), np.shape(ww[rjj-2]), np.shape(ww[rjj-1]), np.shape(re[rjj-1]))
            print(chis[rjj-2], chis[rjj])

            Heff = ncon([le[rjj-2], ww[rjj-2], ww[rjj-1], re[rjj-1]],
            [[-1,1,-5],[1,2,-2,-6],[2,3,-3,-7],[-4,3,-8]])
            
            Heff = Heff.reshape( chis[rjj-2]*dd*dd*chis[rjj], chis[rjj-2]*dd*dd*chis[rjj])
            lam0, eiv0 = LAS.eigsh(Heff, k=1, which='SA',tol=1e-6) #, v0=psi_flat, tol=tol, ncv=N_min)
            u, s, vdag, chiTrunc = mps.SVD_trunc(eiv0.reshape(chis[rjj-2]*dd,dd*chis[rjj]),1e-12,40)
            ss = LA.sqrtm(np.diag(s))
            uss = (u @ ss).reshape(chis[rjj-2],dd,chiTrunc)
            ssv = (ss @ vdag).reshape(chis[rjj],dd,chiTrunc) 

            Apsi[rjj-1] = uss
            Apsi[rjj] = ssv

            # rebuild right env #TODO: do update instead of full rebuild
            re = mpomps.build_right_env(inMPS,inMPO)
            Emin = Emin + lam0