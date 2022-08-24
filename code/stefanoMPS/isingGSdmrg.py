import numpy as np

import matplotlib.pyplot as plt
import myMPSstuff as mps
import myMPOstuff as mpo
import applMPOMPS as mpomps

import myDMRG as dmrg

from isingMPO import IsingMPO
from oned_ising_tenpy import example_DMRG_tf_ising_finite


LLL = 10

# maximum chi 
chiM = 40

gg = 0.4


# Do it with tenpy 
E_tenpy, psi_tenpy, _ = example_DMRG_tf_ising_finite(LLL, gg, chiM)
Smid_tenpy = psi_tenpy.entanglement_entropy()[(LLL-1)//2]



psi = mps.myMPS(mps.randMPS(LLL,chi=20))


Hising = mpo.myMPO(IsingMPO(LLL, J=1., g=gg))

Emin1 = mpomps.expValMPO(psi, Hising)

Emin2 = dmrg.findGS_DMRG(Hising, psi)

print(f"norm3 = {psi.getNorm()}")
Emin3 = mpomps.expValMPO(psi, Hising)
print(f"norm4 = {psi.getNorm()}")

print(Emin1, Emin2, Emin3, E_tenpy)

print(psi.getNorm())