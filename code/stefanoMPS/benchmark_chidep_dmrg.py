import numpy as np

import matplotlib.pyplot as plt
import myMPSstuff as mps
import myMPOstuff as mpo
import myMPOMPS as mpomps

import myDMRG as dmrg

from myIsingMPO import IsingMPO
from oned_ising_tenpy import example_DMRG_tf_ising_finite


from timeit import default_timer as timer
from datetime import timedelta



LLL = 30

gg = 1.1


# Do it with tenpy 

timesTpy = []
chisTpy = [10,20,30,40,50,60]
for i, chim in enumerate(chisTpy):
    start = timer()
    E_tenpy, psi_tenpy, _ = example_DMRG_tf_ising_finite(LLL, gg, chim)
    end = timer()
    timesTpy.append(end-start)

print(chisTpy, timesTpy)


psi = mps.myMPS(mps.randMPS(LLL,chi=20))
Hising = mpo.myMPO(IsingMPO(LLL, J=1., g=gg))
Emin1 = mpomps.expValMPO(psi, Hising)


timesDMRG = []
chisDMRG = [10,20,25,30,35,40]
for i, chim in enumerate(chisDMRG):
    start = timer()
    Emin2 = dmrg.findGS_DMRG(Hising, psi, chiMax = chim, nsweeps = 5)
    end = timer()
    timesDMRG.append(end-start)

print(chisDMRG,timesDMRG)

