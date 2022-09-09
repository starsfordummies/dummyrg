import numpy as np 
import myMPOstuff as mpo
import myMPSstuff as mps
import rotfoldMPO as rf
import myPowerMethod as pm
import myIsingMPO as isi
import myDMRG as dmrg

import matplotlib.pyplot as plt
import gzip, pickle

LL = 10
gz = 1.1    
chiM = 50
iterMax = 3
            
#MPO =  mpo.myMPO(isi.IsingMPO(LL, J=1., g = gz))
MPO = mpo.myMPO(rf.buildRotFoldMPO(2.0, 0.1, gz,  {"mmode": "svd", "ttype": "real", "fold": True}))

ifPsi, iter, ents, devec, energies, chimax_reached = pm.power_method(MPO, 0, chiM, iters=iterMax, full_ents=True)


#psi = mps.myMPS(mps.randMPS(LL,chi=20,d=2))

Emin2, psi = dmrg.findGS_DMRG(MPO, 0, chiMax = chiM, nsweeps = 5, isHermitian=False)


print(energies[-1])
print(Emin2)

print(ents[-1])
print(psi.getEntropies())