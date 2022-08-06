import numpy as np

import matplotlib.pyplot as plt
import myMPSstuff as mps
import myMPOstuff as mpo
import applMPOMPS as mpomps

import myDMRG as dmrg

from isingMPO import IsingMPO
from oned_ising_tenpy import example_DMRG_tf_ising_finite


LLL = 20

# maximum chi 
chiM = 40

gg = 0.4


# Do it with tenpy 
E_tenpy, psi_tenpy, _ = example_DMRG_tf_ising_finite(LLL, gg, chiM)
Smid_tenpy = psi_tenpy.entanglement_entropy()[(LLL-1)//2]



psi = mps.myMPS(mps.randMPS(LLL,chi=20))

Hising = mpo.myMPO(IsingMPO(LLL, J=1., g=gg))

Emin = dmrg.findGS_DMRG(Hising, psi)

Emin = mpomps.expValMPO(psi, Hising)
print(Emin)