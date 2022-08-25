import myMPSstuff as mps 
import myMPOstuff as mpo
import myEnvironmentsMPO as envs

import numpy as np 
from myUtils import sncon as ncon
from scipy import linalg as LA 
import scipy.sparse.linalg as LAS



def findGS_DMRG( inMPO : mpo.myMPO, inMPS: mps.myMPS) -> mps.myMPS:

    if inMPO.DD != inMPS.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")

    if inMPO.LL != inMPS.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
    
    LL = inMPS.LL 

    #if not inMPS.canon: 
    #    print("Bringing to can form first")
    #    inMPS.bringCan(epsTrunc=1e-12)

    inMPS.set_form('R')


    # build environments
    #le, re = envs.build_environments(inMPS, inMPO)

    le = envs.init_env(inMPS.LL)
    re = envs.build_right_env(inMPS, inMPO)

    chis = inMPS.chis
    ww = inMPO.MPO
    psi = inMPS.MPS
    dd = inMPO.DD

    nsweeps = 10

    toleig = 1e-6

    guessTheta = np.random.rand(chis[0]*dd*dd*chis[2])

    for ns in range(0,nsweeps):

        toleig = toleig*0.5
      
        # ===>>===>     (L-R sweep )
        for jj in range(0,LL-1): #, Aj in enumerate(Apsi[:-1]):
            print(f"[L] ncon L({jj}) W({jj}) W({jj+1}) R({jj+2}) updating A[{jj}] B[{jj+1}]")
            #Heff = ncon([le[jj], ww[jj], ww[jj+1], re[jj+2]],
            #[[-1,1,-5],[1,2,-2,-6],[2,3,-3,-7],[-4,3,-8]])

            #Heff = Heff.reshape( chis[jj]*dd*dd*chis[jj+2], chis[jj]*dd*dd*chis[jj+2])


            dimH = chis[jj]*dd*dd*chis[jj+2]
            def HthetaL(th: np.ndarray):
                theta = th.reshape(chis[jj], dd, dd, chis[jj+2])
                #print(np.shape([le[jj-1]), ww[jj-1], theta))
                Lwtheta = ncon([le[jj], ww[jj], theta],[[-1,2,3],[2,-2,-3,4],[3,4,-4,-5]])
                wR = ncon([ww[jj+1],re[jj+2]], [[-2,2,-4,-5],[-1,2,-3]])

                return ncon([Lwtheta,wR], [[-1,2,-2,3,4],[-4,2,4,-3,3]]).reshape(dimH)


            Heff = LAS.LinearOperator((dimH,dimH), matvec=HthetaL)

            lam0, eivec0 = LAS.eigsh(Heff, k=1, which='SA', v0=guessTheta, tol=toleig) #, v0=psi_flat, tol=tol, ncv=N_min)
          
            u, s, vdag, chiTrunc = mps.SVD_trunc(eivec0.reshape(chis[jj]*dd,dd*chis[jj+2]),1e-10,40)
            print(f"[L] truncating {chis[jj]*dd}-{dd*chis[jj+2]} -> {chiTrunc}")
            
            print("Checking unitarity of U: ncon u udag gives")
            print(np.round( np.real_if_close(ncon([u, np.conj(u)],[[1,-1],[1,-2]])), 12 ))
            #ss = LA.sqrtm(np.diag(s))
            ss = np.diag(s) #/ LA.norm(s)

            print(f"Norms: {LA.norm(u)}, {LA.norm(ss)}, {LA.norm(vdag)}")
          
            u = u.reshape(chis[jj],dd,chiTrunc).transpose(0,2,1)
            ssv = (ss @ vdag).reshape(chiTrunc,dd,chis[jj+2]).transpose(0,2,1)


            psi[jj] = u
            psi[jj+1] = ssv  # FIXME: actually unused? except maybe last step? 

            chis[jj+1] = chiTrunc

            if jj < LL-2:
                guessTheta = ncon([ssv,psi[jj+2]],[[-1,1,-2],[1,-4,-3]]) 

            #print(f"truncated {np.shape(Apsi[jj])}, {np.shape(Apsi[jj+1])}")

            print(f"Updated norm: {inMPS.getNorm()}")

            # update left env 
            le = envs.update_left_env(le, u, ww[jj], jj)
 
        print( [np.shape(i) for i in le])
        
        #inMPS.bringCan()
        #inMPS.set_form("L")

        # <====<<====    (R-L sweep)
        Emin = 0.

        envs.update_right_env(re, psi[LL-1], ww[LL-1], LL-1)

        for jj in range(LL-2,0,-1): # in enumerate(Apsi[1:]):
            print(f"[R] ncon L({jj-1}) W({jj-1}) W({jj}) R({jj+1}) updating A[{jj-1}] B[{jj}]")
            #Heff = ncon([le[jj-1], ww[jj-1], ww[jj], re[jj+1]],
            #[[-1,1,-5],[1,2,-2,-6],[2,3,-3,-7],[-4,3,-8]])
            
            #Heff = Heff.reshape( chis[jj-1]*dd*dd*chis[jj+1], chis[jj-1]*dd*dd*chis[jj+1])

            dimH = chis[jj-1]*dd*dd*chis[jj+1]
            def HthetaR(th: np.ndarray) -> np.ndarray:
                theta = th.reshape(chis[jj-1], dd, dd, chis[jj+1])
                #print(np.shape([le[jj-1]), ww[jj-1], theta))
                Lwtheta = ncon([le[jj-1], ww[jj-1], theta],[[-1,2,3],[2,-2,-3,4],[3,4,-4,-5]])
                wR = ncon([ww[jj],re[jj+1]], [[-2,2,-4,-5],[-1,2,-3]])

                return ncon([Lwtheta,wR], [[-1,2,-2,3,4],[-4,2,4,-3,3]]).reshape(dimH)


            Heff = LAS.LinearOperator(shape=(dimH,dimH), matvec=HthetaR)

            lam0, eivec0 = LAS.eigsh(Heff, k=1, which='SA',tol=toleig) #, v0=psi_flat, tol=tol, ncv=N_min)
            u, s, vdag, chiTrunc = mps.SVD_trunc(eivec0.reshape(chis[jj-1]*dd,dd*chis[jj+1]),1e-10,40)

            print(f"[R] truncating {chis[jj-1]*dd} {dd*chis[jj+1]} -> {chiTrunc}")
            #ss = LA.sqrtm(np.diag(s))
            ss = np.diag(s) #/ LA.norm(s)


            uss = (u @ ss).reshape(chis[jj-1],dd,chiTrunc).transpose(0,2,1)  
            vdag = vdag.reshape(chiTrunc,dd,chis[jj+1]).transpose(0,2,1)

            psi[jj-1] = uss
            psi[jj] = vdag
            
            chis[jj] = chiTrunc

            guessTheta = ncon([psi[jj-2],uss],[[-1,1,-2],[1,-4,-3]]) 

            # update right env 
            envs.update_right_env(re, vdag, ww[jj], jj)

            if lam0 < Emin : Emin = lam0
            print(f"{lam0 = }")

        print(f"Nsweep = {ns}, En = {Emin}")
        print(f"chis = {inMPS.chis}")

    return Emin 