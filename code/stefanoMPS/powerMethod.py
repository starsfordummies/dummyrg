
from __future__ import annotations

import applMPOMPS as mpomps
import logging

import myMPSstuff as mps
import myMPOstuff as mpo

# For pretty progress bars 
from tqdm import tqdm  

def power_method(MPO: mpo.myMPO, startMPS: mps.myMPS, chiM: int, iters: int = 200, HMPO: mpo.myMPO | int = 0, full_ents: bool = False): #-> (myMPS, list, list, list, list):

   
    if not isinstance(HMPO, mpo.myMPO):
        print("Returning energies as expectation value of the evolution MPO")
        enMPO = MPO
    else: 
        enMPO = HMPO


    LL = len(startMPS.MPS)

    # Since I have L+1 bonds (including the trivial ones at the beginning/end)
    # the correct index for my midchain bond should be exactly L/2 
    midchain = LL//2

    max_chi_reached = max(startMPS.chis)
    oPsi = mpomps.applyMPOtoMPS(MPO, startMPS)

    # here comes the trunc
    oPsi.bringCan(chiMax = chiM)


    ents = oPsi.getEntropies()
    ent_mid = ents[midchain]

    #delta entropy
    de = 1.

    iter = []
    devec = []
    ent_mids = []
    entropies = []
    energies = []

    progress_bar = tqdm(range(1,iters+1))
    # for j in tqdm(range(1,iters)):
    for jj in progress_bar:
        oPsi = mpomps.applyMPOtoMPS(MPO, oPsi)  
        oPsi.bringCan(chiMax = chiM)

        curr_chi = max(oPsi.chis) 
        if curr_chi > max_chi_reached: max_chi_reached = curr_chi
        progress_bar.set_description(f"{jj}, chi/max {curr_chi}/{max_chi_reached}")

        ents = oPsi.getEntropies(numSVs = chiM)
        ent_new = ents[midchain]
    
        de = abs(ent_mid - ent_new)
        
        #print(f"Entropies: {emid, enew, de}")
        if jj%2 == 0:
            devec.append(de)
            iter.append(jj)
            ent_mids.append(ent_new)
            if full_ents: entropies.append(ents)
            energies.append(mpomps.expValMPO(oPsi,enMPO))
            #print(j, oPsi.chis)


        ent_mid = ent_new

        # maybe break earlier if we converge quickly
        if de < 1e-12:
            print(f"Looks converged after {jj} steps")
            break

    # check that the largest entropy is indeed at midchain
    locMax = ents.index(max(ents)) 
    if locMax != midchain: 
        logging.warning(f"largest entropy is NOT at midchain! ({locMax} vs {midchain})")
        logging.warning(f"Largest entropy: S({locMax}) = {ents[locMax]}")
        logging.warning(f"entropies = {ents}")

    print(f"max chi reached(/max): {max_chi_reached}/{chiM}, final dSmid = {devec[-1]}")

    # return full entropies, not just midchain ones
    if full_ents: 
        print("Returning *ALL* entropies")
        Svec = entropies
    else:
        Svec = ent_mids

    return oPsi, iter, Svec, devec, energies 






def power_method_untilconverged(MPO: mpo.myMPO, startMPS, chiM: int, HMPO: mpo.myMPO | int = 0, full_ents: bool = False, epsConv = 1e-4):
    nSteps = 50
    oPsi, iter, entropies, devec, energies = power_method(MPO, startMPS, chiM, nSteps, HMPO, full_ents)
    nloop = 0 
    while devec[-1] > epsConv:
        nloop += 1 
        print(f"After running {iter[-1]} S is {entropies[-1]}, dS={devec[-1]}, running {nSteps} more iters  ")
        oPsi, temp_iter, temp_entropies, temp_devec, temp_energies = power_method(MPO, oPsi, chiM, nSteps, HMPO, full_ents)
        
        if nloop > 20 or temp_devec[-1] > devec[-1]:
            logging.warning(f"This is likely not converging after {nloop*nSteps} iterations, final dSmin = {devec[-1]}")
        iter.extend([ii + iter[-1] for ii in temp_iter])
        entropies.extend(temp_entropies)
        devec.extend(temp_devec)
        energies.extend(temp_energies)
  
       

    print(f"Converged after {iter[-1]} steps: deltaS = {devec[-1]}")

    return oPsi, iter, entropies, devec, energies 
