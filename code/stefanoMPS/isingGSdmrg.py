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


psi = mps.myMPS(mps.randMPS(LLL))

Hising = mpo.myMPO(IsingMPO(LLL, J=1., g=gg))

Emin = dmrg.findGS_DMRG(Hising, psi)

print(Emin)