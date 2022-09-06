"""Example illustrating the use of DMRG in tenpy.

The example functions in this class do the same as the ones in `toycodes/d_dmrg.py`,
but make use of the classes defined in tenpy.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
#from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg



def example_DMRG_tf_ising_finite(L, g, chimax = 50) -> tuple[float, MPS, TFIChain]:
    print("finite DMRG, transverse field Ising model")
    print("L={L:d}, g={g:.2f}".format(L=L, g=g))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': chimax,
            'svd_min': 1.e-12
        },
        'combine': True
    }
    info = dmrg.run(psi, M, dmrg_params)  # the main work...
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    # mag_x = np.sum(psi.expectation_value("Sigmax"))
    # mag_z = np.sum(psi.expectation_value("Sigmaz"))
    # print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    # print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))

    print(f"midchainEE: {psi.entanglement_entropy()[(L-1)//2]} ")

    """
    if L < 20:  # compare to exact result
        from tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
        print("relative error: ", abs((E - E_exact) / E_exact))
    """

    return E, psi, M




def example_1site_DMRG_tf_ising_finite(L, g, chimax = 50):
    print("single-site finite DMRG, transverse field Ising model")
    print("L={L:d}, g={g:.2f}".format(L=L, g=g))
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': chimax,
            'svd_min': 1.e-12
        },
        'combine': False,
        'active_sites': 1  # specifies single-site
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    # mag_x = np.sum(psi.expectation_value("Sigmax"))
    # mag_z = np.sum(psi.expectation_value("Sigmaz"))
    #print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    #print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))

    #print(f"EE = {psi.entanglement_entropy()}")
    print(f"midchainEE: {psi.entanglement_entropy()[(L-1)//2]} ")

    """
    if L < 20:  # compare to exact result
        from tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
        print("relative error: ", abs((E - E_exact) / E_exact))
    """
    return E, psi, M





if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    example_DMRG_tf_ising_finite(L=30, g=0.9)
    print("-" * 100)
    example_1site_DMRG_tf_ising_finite(L=30, g=0.9)
    print("-" * 100)
  
