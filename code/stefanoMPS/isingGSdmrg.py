import numpy as np

import matplotlib.pyplot as plt
import myMPSstuff as mps
import myMPOstuff as mpo
import myMPOMPS as mpomps

import myDMRG as dmrg

from myIsingMPO import IsingMPO
from oned_ising_tenpy import example_DMRG_tf_ising_finite

import cProfile
import pstats


LLL = 20

# maximum chi 
chiM = 50

gg = 0.9


# Do it with tenpy 
E_tenpy, psi_tenpy, _ = example_DMRG_tf_ising_finite(LLL, gg, chiM)
Smid_tenpy = psi_tenpy.entanglement_entropy()[(LLL-1)//2]



psi = mps.myMPS(mps.randMPS(LLL,chi=20))

Hising = mpo.myMPO(IsingMPO(LLL, J=1., g=gg))

Emin1 = mpomps.expValMPO(psi, Hising)


# with cProfile.Profile() as pr:
    
Emin2 = dmrg.findGS_DMRG(Hising, psi, 50, 5)

Emin3 = mpomps.expValMPO(psi, Hising)

print(f"Before DMRG: {Emin1} \n After DMRG: {Emin2} \n DMRG+Norm: {Emin3} \n Tenpy: {E_tenpy}")
print(f"EE {Smid_tenpy = }, {psi.getEntropies()[(LLL-1)//2] = }")

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# #stats.print_stats()
# stats.dump_stats(filename="profi_ising.prof")
