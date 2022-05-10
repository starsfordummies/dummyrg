#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last modified: 2022/05/04 14:13:57

Created on Wed Feb 10 12:35:12 2021

@author: luca, stefano
"""
import sys

import numpy as np
#from scipy.sparse.linalg import eigsh
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

from tenpy.tools import process

import copy
from model_isingselfdual import TFISDChain


import argparse

HIMEM = 8000.

memFactor = 1.
if sys.platform == u'darwin':
    # On OSX the result is in bytes.
    print("Running on MacOS")
    memFactor = 1000.


def find_ground_state(L,theta0,chi,dmrg_params):
    model_params = dict(
            L=L,
            J=0.1,
            g=-np.cos(theta0)/np.sin(theta0),
            bc_MPS='finite',
            conserve='best'
            )

    # model_params = {
    #     'L': L,
    #     'S': 0.5,
    #     'conserve': 'best',
    #     'Jz': 0.,
    #     'Jy': 0.,
    #     'Jx': -1.,
    #     'hx': 0.0,
    #     'hy': 0.0,
    #     'hz': -np.cos(theta0)/np.sin(theta0),
    #     'muJ': 0.0,
    #     'bc_MPS': 'finite',
    # }

    model0 = TFISDChain(model_params)
    sites = model0.lat.mps_sites()
    psi = MPS.from_product_state(sites, ['up'] * (L), "finite")
    info = dmrg.run(psi, model0, dmrg_params)
    E = info['E']
    #print("E = {E:.13f}".format(E=E))
    print(f"{E = :.13f}")
    print("final bond dimensions: ", psi.chi)

    return psi, model0






def main_quench():
    parser = argparse.ArgumentParser()
    parser.add_argument("L", help="chain length", type=int)
    parser.add_argument("th0", help="mag field, theta0 = pi/arg", type=float)
    parser.add_argument("max_chi", help="max bond dimension", type=int)
    #parser.add_argument('timesL', nargs='?', default='1', type=int)
    parser.add_argument("-tL", "--timesL", default = 1, type=int, help="how many times L we do the evolution")
    parser.add_argument("-r", "--resume", help="continue from previous", action="store_true")
    parser.add_argument("-h5", "--hdf5", help="use h5 instead of pickle", action="store_true")
    parser.add_argument("-wt", "--write_temp", help="store temp files every X iterations", type=int, default=0)
    
    args = parser.parse_args()
   

    #print("Arguments: [1]L, [2]theta0=np.pi/arg, [3]max_chi, [4]continue_from_prev")

    L = args.L
    theta0 = np.pi/args.th0
    thetac = np.pi/4  #critical theta
    max_chi = args.max_chi

    suffix = ".pkl"

    if args.hdf5:
       import h5py
       suffix = ".h5"
    else:
        import gzip 
        import pickle

    from tenpy.tools import hdf5_io

    # DMRG and TDVP parameters 
    dmrg_params = {
        'trunc_params': {
             'chi_max': max_chi,
             'svd_min': 1.e-10,
            #'trunc_cut': None
          },
        #'update_env': 20,
        #'start_env': 20,
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-10,
        'mixer': False 
    }

    timesL = args.timesL
    tmax = timesL*L

    delta_t=0.1

    tdvp_params = {
        'start_time': 0.,
        'dt': delta_t,
        'trunc_params': {
            'chi_max': max_chi,
            'svd_min': 1.e-10,
            'degeneracy_tol': 1.e-10,
            'trunc_cut':  1.e-10,
            #'verbose': 0
            }
        }

    if max_chi > 2**(L/2):
        print(f"With this {L = } and {max_chi = } > {2**(L/2) = }")
        print("The result should be exact (not limited by bond dimension)")
        #exit()

    print("##################################################")
    print(f"Chain length {L}")
    print(f"quench from theta0=pi/{args.th0} = {np.pi/float(args.th0)} to pi/4 ={np.pi/4}")
    print(f"that is, cot(theta) -> cot(theta0) = {np.cos(theta0)/np.sin(theta0):.4f} -> {np.cos(np.pi/4)/np.sin(np.pi/4):.4f}")
    print(f"Evolve until {tmax = } at steps {delta_t = }")
    print(f"max chi: {max_chi}")
    if args.resume:
        print(f"continue from previous")
    if args.hdf5:
        print("Using h5 output:")
        print(h5py.version.info)
    else:
        print("Using pickle output")
    if args.write_temp:
        print(f"Saving temp data files every {args.write_temp} steps")

    print("##################################################")

    data_path= 'DATA/'
  
    # Data filename for quench 
    name_file= 'TDVP_ising'
    name_file= data_path+name_file + '_L_' + str(L) + '_theta_pi4_from_theta0_pi'+str(args.th0) + \
            '_max_chi_'+str(max_chi)+'_symm_'

    
    model_params = dict(L=L, J=0.1, g=-np.cos(thetac)/np.sin(thetac), bc_MPS='finite', conserve='best')
    model = TFISDChain(model_params)


    #model = Potts_symm({"L": L, "J": - 1, "f": - np.cos(thetac)/np.sin(thetac), "bc_MPS": "finite"})

    
    # If we start from the beginning, calculate GS at theta0 first
    if args.resume:

        print(f'Loading from {name_file}one_site_temp{suffix}')
        if args.hdf5:
            with h5py.File(name_file+'one_site_temp'+suffix, 'r') as f:
                dataTDVP = hdf5_io.load_from_hdf5(f)
        else:
            with gzip.open(name_file+'one_site_temp'+suffix, 'rb') as f:
                dataTDVP = pickle.load(f)

        psi=dataTDVP.get('psi')
        times=dataTDVP.get('times')
        S_all=dataTDVP.get('entropies')
        ent_spect=dataTDVP.get('spectrum')
        energies=dataTDVP.get('energies')
        chi_t=dataTDVP.get('chi_t')
        max_chi_t=max_chi+1  # ??? 


        tdvp_params["start_time"] = times[-1]

        print(f"Data loaded, starting from time {times[-1]}")

    else: # No previous data, run DMRG + TWO-STEP evolution
      
        print("Finding ground state with DMRG first ") 
        psi, model0 = find_ground_state(L,theta0,max_chi,dmrg_params)

        dataDMRG = {
            "psi": psi, 
            "model": model0,
            "dmrg_params": dmrg_params,
            "parameters": {"L": L, "theta": theta0},
            "chi_t": max_chi
        }

        name_file_gs= 'ground_state_ising'
    
        name_file_gs = data_path + name_file_gs + '_L_' + str(L) + '_theta_pi' + str(args.th0)+\
            '_max_chi_'+str(max_chi)+'_symm_'

        print(f'Saving GS info in {name_file_gs}{suffix}') 
        if args.hdf5:
            with h5py.File(name_file_gs+suffix, 'w') as f:
                hdf5_io.save_to_hdf5(f, dataDMRG)
        else:
            with gzip.open(name_file_gs+suffix, 'wb') as f:
                pickle.dump(dataDMRG, f)

        print(f'After ground state for theta = pi/{args.th0}, do the quench')


        '''#TODO: this should not be necessary really, we already have psi..
        if args.hdf5:
            print(f'Loading GS psi from {name_file_gs}.h5')
            with h5py.File(name_file_gs+'.h5', 'r') as f:
                dataDMRG = hdf5_io.load_from_hdf5(f)
        else:
            with gzip.open(name_file_gs+'.pkl', 'rb') as f:
                dataDMRG = pickle.load(f)
        psi=dataDMRG.get('psi')
        '''

        times = []
        S_all = []
        ent_spect =[]
        energies =[]
        chi_t=[]
        max_chi_t=1
        i=0


        # Clean up some unneeded stuff
        dataDMRG.clear()

    

        tdvp_engine = tdvp.TDVPEngine(psi, model, tdvp_params)

        # The main two-site TDVP loop 
        while max_chi_t < max_chi and tdvp_engine.evolved_time <= tmax :
            
            tdvp_engine.run_two_sites(N_steps=1)

            psi_2=copy.deepcopy(psi)
            psi_2.canonical_form()

            times.append(tdvp_engine.evolved_time)
            S_all.append(psi_2.entanglement_entropy())
            ent_spect.append(psi_2.entanglement_spectrum())
            energy_t=np.real_if_close(model.H_MPO.expectation_value(psi_2),tol=10000)
            energies.append(energy_t)
            max_chi_t=np.max(psi.chi)
            chi_t.append(max_chi_t)

            i=i+1
            print(f'# [2S] Iteration {i} Energy {energy_t}  chiMax {max_chi_t} times {times[-1]:.1f}'  )


        dataTDVP = {
            "psi": psi,  
            "model": model,
            "tdvp": tdvp_params,
            "parameters": {"L": L, "theta": thetac},
            "times": times,
            "entropies": S_all,
            "spectrum": ent_spect,
            "energies": energies,
            "chi_t": chi_t
        }



    ### ONE STEP EVOLUTION


    nStepsOneS = int((tmax-times[-1])/delta_t)
    if nStepsOneS < 1: nStepsOneS = 1
    print(f"## Maxed out chi at time {times[-1]}, running {nStepsOneS} one-site updates")

    for i in range(nStepsOneS):  #was: 200
          tdvp_engine.run_one_site(N_steps=1)
          #psi.canonical_form()
          times.append(tdvp_engine.evolved_time)

          psi_2=copy.deepcopy(psi)
          psi_2.canonical_form()
          S_all.append(psi_2.entanglement_entropy())
          ent_spect.append(psi_2.entanglement_spectrum())

          energy_t=np.real_if_close(model.H_MPO.expectation_value(psi_2),tol=10000)
          energies.append(np.real(energy_t))

          #max_chi_t=np.max(psi.chi)  # Doesn't get changed anymore in oneSite algo
          #chi_t.append(max_chi_t)

          
          #print('#### [1S] Iteration n '+str(i)+' Energy '+ str(energy_t) + '  chiMax ' + str(max_chi_t)  )
          print(f'#[1S] Iteration {i}/{nStepsOneS} Energy {energy_t}  chiMax {max_chi_t} times {times[-1]:.1f}'  )
          
          memusage = process.memory_usage()/memFactor

          if np.mod(i,20)==0:
            print(f"Mem usage: {memusage:.1f} MB")

          if memusage > HIMEM or (args.write_temp and np.mod(i,args.write_temp)==0):
            if args.hdf5:
                print(f'######  Writing checkpoint in  {name_file}one_site_temp{suffix}'  )
                with h5py.File(name_file+'one_site_temp'+suffix, 'w') as f:
                    hdf5_io.save_to_hdf5(f, dataTDVP)
            else:
                print(f'######  Writing checkpoint in  {name_file}one_site_temp{suffix}'  )
                with gzip.open(name_file+'one_site_temp'+suffix, 'wb') as f:
                    pickle.dump(dataTDVP, f)

          if memusage > HIMEM:
              print("High memory usage! Writing down temp file and exit")
              exit()

    # End of the iterations 
    print(f'\n\n###################################################'  )
    print(f'# Reached end, Writing to {name_file}one_site{suffix}'  )

    if args.hdf5:
        with h5py.File(name_file+'one_site'+suffix, 'w') as f:
            hdf5_io.save_to_hdf5(f, dataTDVP)
    else:
        with gzip.open(name_file+'one_site'+suffix, 'wb') as f:
            pickle.dump(dataTDVP, f)

    print("Parameters used:")
    print(args)
    print('\n\n')

    #return ent_spect


if __name__ == '__main__':
    import logging
    #logging.basicConfig(level=logging.INFO)
    main_quench()
