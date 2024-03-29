import myMPSstuff as mps 
import myMPOstuff as mpo
import myEnvironmentsMPO as envs

import numpy as np 
from tqdm import tqdm 

from myUtils import sncon as ncon

from scipy import linalg as LA 
import scipy.sparse.linalg as LAS

import functools



def findGS_DMRG( inMPO : mpo.myMPO, inMPS: any = 0, chiMax: int = 50, nsweeps: int = 5, isHermitian="True") -> complex:
    """ My DMRG algo, modifies in-place psi, returns the energy """

    if isHermitian:
        print("Assuming HERMITIAN Ham")
    else:
        print("Assuming NON-Hermitian Ham")

    # if we don't input a starting MPS, generate a random one
    if not isinstance(inMPS, mps.myMPS):
        inMPS = mps.myMPS(mps.randMPS(LL=inMPO.LL, chi=10, d=inMPO.DD))


    if inMPO.DD != inMPS.DD: 
        raise ValueError(f"MPO and MPS don't have the same physical dimension (D={inMPO.DD} vs {inMPS.DD}),  aborting")

    if inMPO.LL != inMPS.LL: 
        raise ValueError(f"MPO and MPS don't have the same length (L={inMPO.LL} vs {inMPS.LL}),  aborting")
    
    LL = inMPO.LL
    ww = inMPO.MPO
    dd = inMPO.DD
  
    # Intialize envs, we start with a L->R sweep so we want to start in right-canonical form 
    # and build the right environment (we'll build the left along the way)

    inMPS.set_form('R')

    le = envs.init_env(inMPS.LL)
    re = envs.build_right_env(inMPS, inMPO)

    chis = inMPS.chis


    # TODO: convert psi to working indices here and switch back at the end to the (vL,vR,d) convention?
    psi = inMPS.MPS
    SVs = inMPS.SV

    toleig = 1e-2
    tolSVD = 1e-4

    Emin_prev = 1e10
    midentr_prev = 1e10


    def Htheta_contract(theta, l,wl,wr,r):
        theta = theta.reshape(np.shape(l)[0], dd, dd, np.shape(r)[0])
        tlist = [l,  wl,   theta,    wr,    r]
        clist =[ [-1, 3, 2], [3, 1, -2, 4], [2, 4, 5, 7], [1, 6, -3, 5], [-4, 6, 7]]
        return ncon(tlist, clist).reshape(dimH)


   
    for ns in range(0,nsweeps):

        # As we go to later sweeps, progressively increase the precision in the eigensolver
        toleig = toleig*0.1
        tolSVD = tolSVD*0.1
      
        # >>>>>>>>>>> (L-R sweep ) >>>>>>>>>>>>>
        progress_bar = tqdm(range(0,LL-1), ascii=" >=")
        progress_bar.set_description(f"{ns}L>")
        for jj in progress_bar:
     
            #print(f"[L] updating A[{jj}] - chi[{jj+1}] - (B[{jj+1}])")

            dimH = chis[jj]*dd*dd*chis[jj+2]


            HthetaL = functools.partial(Htheta_contract, l=le[jj],  wl=ww[jj], wr=ww[jj+1], r=re[jj+2])
            
            Heff = LAS.LinearOperator(shape=(dimH,dimH), matvec=HthetaL)

            guessTheta = ncon([psi[jj],psi[jj+1]],[[-1,1,-2],[1,-4,-3]]) 

            if isHermitian:
                lam0, eivec0 = LAS.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig) 
            else:
                lam0, eivec0 = LAS.eigs(Heff, k=1, which='SR', v0=guessTheta , tol=toleig) 


          
            u, s, vdag, chiTrunc = mps.SVD_trunc(eivec0.reshape(chis[jj]*dd,dd*chis[jj+2]),tolSVD,chiMax)
          
            sn = s / LA.norm(s)
            ss = np.diag(sn) 

            ur = u.reshape(chis[jj],dd,chiTrunc).transpose(0,2,1)
            ssv = (ss @ vdag).reshape(chiTrunc,dd,chis[jj+2]).transpose(0,2,1)


            psi[jj] = ur
            chis[jj+1] = chiTrunc
            psi[jj+1] = ssv

            # It shouldn't be necessary to update the SVs here (except for the last link),
            # we'll do it in the right sweep

            if jj == LL-2:  # reached the edge: jj = L-2 - start going backwards
                SVs[jj+1] = sn
                #print(f"updating B[{jj+1}]")
                psi[jj+1] = vdag.reshape(chiTrunc,dd,chis[jj+2]).transpose(0,2,1)

            # update left env 
            envs.update_left_env(le, psi[jj], ww[jj], jj)
 
        # end left sweep


       
        Emin = 0.

        envs.update_right_env(re, psi[LL-1], ww[LL-1], LL-1)

        # TODO CHECK: end at 1 (so 2) or 0 (so ends at 1 ?)
        progress_bar = tqdm(range(LL-2,0,-1), ascii=' <=')

        progress_bar.set_description(f"{ns}<R [{max(inMPS.chis)}]")

        # <<<<<<<<<<<<<  (R-L sweep) <<<<<<<<<<<<<<<<
        for jj in progress_bar:
            #print(f"[R] updating (A[{jj-1}]) - chi[{jj}] - B[{jj}]")
        
            dimH = chis[jj-1]*dd*dd*chis[jj+1]


            HthetaR = functools.partial(Htheta_contract, l=le[jj-1],  wl=ww[jj-1], wr=ww[jj], r=re[jj+1])
            
            Heff = LAS.LinearOperator(shape=(dimH,dimH), matvec=HthetaR)

            guessTheta = ncon([psi[jj-1],psi[jj]],[[-1,1,-2],[1,-4,-3]]) 

            if isHermitian:
                lam0, eivec0 = LAS.eigsh(Heff, k=1, which='SA', v0=guessTheta , tol=toleig) 
            else:
                lam0, eivec0 = LAS.eigs(Heff, k=1, which='SR', v0=guessTheta , tol=toleig) 

            u, s, vdag, chiTrunc = mps.SVD_trunc(eivec0.reshape(chis[jj-1]*dd,dd*chis[jj+1]),tolSVD, chiMax)

            sn = s / LA.norm(s)
            ss = np.diag(sn) 

            uss = (u @ ss).reshape(chis[jj-1],dd,chiTrunc).transpose(0,2,1)  
            vdag = vdag.reshape(chiTrunc,dd,chis[jj+1]).transpose(0,2,1)

            psi[jj-1] = uss 
            SVs[jj] = sn
            psi[jj] = vdag
            chis[jj] = chiTrunc

            
            if jj == 1:  # reached the edge: jj = 1 - start going forward
                #print(f"Updating A[{jj-1}]")
                psi[jj-1] = u.reshape(chis[jj-1],dd,chiTrunc).transpose(0,2,1)


            # update right env 
            envs.update_right_env(re, psi[jj], ww[jj], jj)

            if lam0 < Emin : Emin = lam0
            #print(f"{lam0 = }")

            # end right sweep
       
        #print(f"chis = {inMPS.chis}")

        midentr = inMPS.getEntropies(checkCan = False)[LL//2]
     
      
        if(abs(Emin - Emin_prev) < 1e-13):
            print(f"Converged after {ns+1} sweeps")
            print(f"deltaS = {midentr - midentr_prev}")
            break
        else: 
            print(f"Nsweep = {ns}, En = {Emin}, deltaE = {abs(Emin - Emin_prev)}, deltaSmid = {midentr - midentr_prev}")
            Emin_prev = Emin

        midentr_prev = midentr

    # Do a final SVD to make the MPS fully right-canonical 
    utail, stail, vdag = LA.svd( (u@ss).reshape(chis[0],dd*chiTrunc), full_matrices=False)
    psi[0] = vdag.reshape(chis[0],dd,chiTrunc).transpose(0,2,1)

    #print(f"Final B0: {psi[0]}")
    #print(f"Final ncon (is it B norm?) {ncon([psi[0],np.conj(psi[0])],[[-1,1,2],[-2,1,2]])}")

    print(f"Final norm: {LA.norm(utail@stail)}")
    #print(inMPS.checkNormalized())
    #print(f"{ inMPS.checkSVsAreOne() = }")
    #print(ccan(psi,0))

    return Emin, inMPS





if __name__ == "__main__":

    from myIsingMPO import IsingMPO

    from timeit import default_timer as timer



    LLL=20
    gg=0.9

    Hising = mpo.myMPO(IsingMPO(LLL, J=1., g=gg))

    start = timer()
    Emin2 = findGS_DMRG(Hising, 0 , chiMax = 100, nsweeps = 3)
    end = timer()
    print(f"time: {end-start}")
