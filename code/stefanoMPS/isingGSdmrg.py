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

from timeit import default_timer as timer
from datetime import timedelta



LLL = 30

# maximum chi 
chiM = 100

gg = 1.1


# Do it with tenpy 

start = timer()
E_tenpy, psi_tenpy, _ = example_DMRG_tf_ising_finite(LLL, gg, chiM)
S_tenpy = psi_tenpy.entanglement_entropy()
end = timer()

print(timedelta(seconds=end-start))


psi = mps.myMPS(mps.randMPS(LLL,chi=20))

Hising = mpo.myMPO(IsingMPO(LLL, J=1., g=gg))

Emin1 = mpomps.expValMPO(psi, Hising)

doprofile = False

if doprofile:
    with cProfile.Profile() as pr:

        start = timer()
        Emin2 = dmrg.findGS_DMRG(Hising, psi, chiMax = chiM, nsweeps = 5)
        end = timer()

else: 

    start = timer()
    Emin2 = dmrg.findGS_DMRG(Hising, psi, chiMax = chiM, nsweeps = 5)
    end = timer()

print(timedelta(seconds=end-start))
sAfterDMRG = psi.getEntropies()

Emin3 = mpomps.expValMPO(psi, Hising)

print(f"Before DMRG: {Emin1} \n After DMRG: {Emin2} \n DMRG+Norm: {Emin3} \n Tenpy: {E_tenpy}")
print(f"EE tenpy:\n {S_tenpy}\n after DMRG: \n {sAfterDMRG}")

if doprofile:
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    #stats.print_stats()
    stats.dump_stats(filename="profi_ising.prof")
