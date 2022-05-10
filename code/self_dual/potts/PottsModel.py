#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:09:35 2021

@author: niallrobertson
"""

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.lattice import Site, Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import asConfig
import math
import numpy as np


class Potts_symm(CouplingMPOModel, NearestNeighborModel):
    """Spin-1 Potts_model with Z3 conservation.
    The Hamiltonian reads:
    .. math ::
        H = \sum_i \mathtt{J} (\tau^{\dagger}_i \tau_{i+1} + h.c)
                 + \mathtt{f} \sigma_i+\sigma_i^{\dagger}
    
    where the matrices are defined as
    \sigma = 0, 1, 0
             0, 0, 1
             1, 0, 0
    \tau =  1,  0,    0 
            0, \omega, 0
            0,  0,    \omega^2
            
     \omega= exp(i 2 Pi/ 3) see Fendely 2012 eq. 16 where we have set \phi = 0
       
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.
    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`XXZChain` below.
    Options
    -------
    .. cfg:config :: XXZChain
        :include: CouplingMPOModel
        L : int
            Length of the chain.
        Jxx, Jz, hz : float | array
            Coupling as defined for the Hamiltonian above.
        bc_MPS : {'finite' | 'infinte'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """
    def __init__(self, model_params):
        model_params = asConfig(model_params, "Potts")
        model_params.setdefault('lattice', "Chain")
        model_params.setdefault('explicit_plus_hc', "True")
    
        CouplingMPOModel.__init__(self, model_params)
    
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'z3')
        omega = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0) )
        omega_conj = complex(math.cos(2.0 * math.pi / 3.0), -math.sin(2.0 * math.pi / 3.0) )
        chinfo = npc.ChargeInfo([3], ['Z3'])
        
        #leg = npc.LegCharge.from_trivial(3)
        leg = npc.LegCharge.from_qflat(chinfo, [[0],[1],[2]])  # charges for up, down
        tau =  [[0., 1., 0.],
                [0, 0, 1.],
                [1., 0, 0]]
        
        tau_d =  [[0, 0, 1.],
                  [1., 0, 0],
                  [0, 1., 0]]
       
       # tau_plus_td = np.add(tau , tau_d)
       #incompatible with z3 symmetry on site
        sigma = [[1.,  0,          0],
                 [0,  omega,      0],
                 [0,  0, omega ** 2]]
        
        sigma_conj = [[1.,   0,            0],
                      [0,   omega_conj,   0],
                      [0,   0,            omega_conj ** 2]]
        
        sigma_plus_sigma_conj = np.add(sigma ,sigma_conj)
        site = Site(leg, ['0','1','2'], sigma = sigma,sigma_conj = sigma_conj,\
                          tau= tau, tau_d = tau_d, sigma_plus_sigma_conj = sigma_plus_sigma_conj)
        #I acutally think that returning something in 
        return site
    
    def init_terms(self, model_params):       
        J = model_params.get('J', 1)
        f = model_params.get('f', 1)

        for u in range(len(self.lat.unit_cell)):  
            self.add_onsite(f, u, 'sigma_plus_sigma_conj')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J, u1, 'tau', u2, 'tau_d', dx, plus_hc=True)

class Potts(CouplingMPOModel, NearestNeighborModel):
    def __init__(self, model_params):
        model_params = asConfig(model_params, "Potts")
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)
    
    def init_sites(self, model_params):
        omega = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0) )
        omega_conj = complex(math.cos(2.0 * math.pi / 3.0), -math.sin(2.0 * math.pi / 3.0) )
        leg = npc.LegCharge.from_trivial(3)
        t_plus_td = [[0,1,1],[1,0,1],[1,1,0]]
        sigma = [[1, 0, 0],[0, omega, 0],[0 ,0, omega ** 2]]
        sigma_conj = [[1, 0, 0],[0, omega_conj, 0],[0 ,0, omega_conj ** 2]]
        site = Site(leg, ['1','0','-1'], sigma = sigma,  sigma_conj = sigma_conj, t_plus_td = t_plus_td)
        
        return site
        
    def init_terms(self, model_params):       
        J = model_params.get('J', 1)
        f = model_params.get('f', 1)

        for u in range(len(self.lat.unit_cell)):  
            self.add_onsite(f, u, 't_plus_td')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J, u1, 'sigma', u2, 'sigma_conj', dx, plus_hc=True)
            

class Potts_Fixed(Potts):        
    def init_terms(self, model_params):
        L = model_params.get('L', 4)
        J = model_params.get('J', 1)
        f = model_params.get('f', 1)
        
        Jbulk = np.asarray([J,] + (L-2) * [J])
        fbulk = np.asarray([0,] + (L-1) * [f])
        Jbdry = np.asarray([0] + [J] + (L-2) * [0])
    
        
        
        for u in range(len(self.lat.unit_cell)):  
            self.add_onsite(fbulk, u, 't_plus_td')
                        
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jbulk, u1, 'sigma', u2, 'sigma_conj', dx, plus_hc=True)
            
# =============================================================================
#         for u in range(len(self.lat.unit_cell)):  
#             self.add_onsite(Jbdry, u, 'sigma')
#             
#         for u in range(len(self.lat.unit_cell)):  
#             self.add_onsite(Jbdry, u, 'sigma_conj')
#             
# =============================================================================

class Potts_Fixed_single(Potts): 
    def init_sites(self, model_params):
        omega = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0) )
        omega_conj = complex(math.cos(2.0 * math.pi / 3.0), -math.sin(2.0 * math.pi / 3.0) )
        leg = npc.LegCharge.from_trivial(3)
        t_plus_td = [[0,1,1],[1,0,1],[1,1,0]]
        sigma = [[1, 0, 0],[0, omega, 0],[0 ,0, omega ** 2]]
        sigma_conj = [[1, 0, 0],[0, omega_conj, 0],[0 ,0, omega_conj ** 2]]
        #p = [[1,0,0],[0,0,0],[0,0,0]]
        site = Site(leg, ['1','0','-1'], sigma = sigma,  sigma_conj = sigma_conj, t_plus_td = t_plus_td)
        
        return site
        
    def init_terms(self, model_params):
        L = model_params.get('L', 4)
        J = model_params.get('J', 1)
        f = model_params.get('f', 1)
        
        #Jbulk = np.asarray([0,] + (L-1) * [J])
        #fbulk = np.asarray([0,] + (L-1) * [f])
        Jbdry = np.asarray([J,] + (L-1) * [0])
    
        for u in range(len(self.lat.unit_cell)):  
            self.add_onsite(f, u, 't_plus_td')
          
        for u in range(len(self.lat.unit_cell)):  
            self.add_onsite(Jbdry, u, 'sigma')
            self.add_onsite(Jbdry, u, 'sigma_conj')
            
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J, u1, 'sigma', u2, 'sigma_conj', dx, plus_hc=True)
            
# =============================================================================
#         for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
#             self.add_coupling(Jbdry, u1, 'p', u2, 'sigma_conj', dx, plus_hc=True)
# =============================================================================

class Potts_Fixed_symm(Potts_symm):        
    def init_terms(self, model_params):
        L = model_params.get('L', 4)
        J = model_params.get('J', 1)
        f = model_params.get('f', 1)
        
        Jbulk = np.asarray([J,] + (L-2) * [J])
        fbulk = np.asarray([0,] + (L-1) * [f])
        Jbdry = np.asarray([0] + [J] + (L-2) * [0])
    
        for u in range(len(self.lat.unit_cell)):  
            self.add_onsite(fbulk, u, 'sigma_plus_sigma_conj')
                        
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jbulk, u1, 'tau', u2, 'tau_d', dx, plus_hc=True)

