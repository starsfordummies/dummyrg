
from __future__ import annotations

from applMPOMPS import applyMPOtoMPS
import logging

#logging.basicConfig()
#logger = logging.getLogger('Something')
#logger.setLevel(logging.DEBUG)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from myMPOstuff import myMPO

# For pretty progress bars 
from tqdm import tqdm  

def power_method(MPO: myMPO, startMPS, chiM: int, iters: int = 200, HMPO: myMPO = 0, full_ents: bool = False):

    if HMPO == 0:
        print("Returning energies as expectation value of the evolution MPO")
        HMPO = MPO

    LL = len(startMPS.MPS)

    # Since I have L+1 bonds (including the trivial ones at the beginning/end)
    # the correct index for my midchain bond should be exactly L/2 
    midchain = LL//2

    max_chi_reached = max(startMPS.chis)
    oPsi = applyMPOtoMPS(MPO, startMPS)

    # here comes the trunc
    oPsi.bringCan(chiMax = chiM)


    ents = oPsi.getEntropies()
    ent_mid = ents[midchain]

    de = 1.

    iter = [0.]
    devec = [float("NaN")]
    ent_mids = [ent_mid]
    entropies = [ents]
    energies = [oPsi.expValMPO(HMPO)]


    progress_bar = tqdm(range(1,iters+1))
    # for j in tqdm(range(1,iters)):
    for jj in progress_bar:
        oPsi = applyMPOtoMPS(MPO, oPsi)  
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
            energies.append(oPsi.expValMPO(HMPO))
            #print(j, oPsi.chis)


        ent_mid = ent_new

    # check that the largest entropy is indeed at midchain
    locMax = ents.index(max(ents)) 
    if locMax != midchain: 
        logging.error(f"largest entropy is NOT at midchain! ({locMax} vs {midchain})")
        logging.error(f"Largest entropy: S({locMax}) = {ents[locMax]}")
        logging.error(f"entropies = {ents}")

    print(f"max chi reached: {max_chi_reached} (vs. chiMax = {chiM})")

    # return full entropies, not just midchain ones
    if full_ents == False:  entropies = ent_mids

    return oPsi, iter, entropies, devec, energies 
