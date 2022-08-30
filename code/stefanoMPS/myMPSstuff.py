# Last modified: 2022/08/30 15:41:06

from __future__ import annotations

import numpy as np
#from numpy import linalg as LA
from numpy.linalg import qr 
from myUtils import robust_svd as rsvd
from myUtils import sncon as ncon
from myUtils import real_close as rc
import logging

# MPS Indices convention:
# v_L , v_R , phys 


def randMPS(LL: int=10, chi: int=5, d: int=2) -> list[np.ndarray]:

    """ Random MPS matrix list Builder function - OBC for now 
    Returns a list of length LL of random tensors with bond dimension chi 
    and physical dimension d 
    """

    logging.info(f"Building random MPS with length {LL}, chi = {chi} and physical d={d}")

    # Build the first element of the MPS A_1 (actually 0th elem of the list)
    outMPS = [ np.random.rand(1,chi,d) + 1j*np.random.rand(1,chi,d) ]

    #Build all the others, A_2 to A_(N-1)
    for _ in range(2,LL):  # 2 to N-1, so actually building N-2 elements
         outMPS.append(np.random.rand(chi,chi,d) + 1j*np.random.rand(chi,chi,d) )
 
    outMPS.append( np.random.rand(chi,1,d) + 1j*np.random.rand(chi,1,d))

    return outMPS


def plusState(LL: int=10) -> list[np.ndarray]:

    """ Returns a product "plus" state
    """

    chi =1 
    d = 2
    logging.info(f"Building product |+> state of length {LL} (physical d={d})")

    # Build the first element of the MPS A_1 (actually 0th elem of the list)
    plus = [1./np.sqrt(d)]*d
    outMPS = [np.array(plus).reshape(chi,chi,d)]*LL

    return outMPS



def bigEntState(LL: int=10) -> list[np.ndarray]:
    #TODO: implement a state with large entropy 
    chi = 2
    d = 2
    logging.info(f"Building product |+> state of length {LL} (physical d={d})")

    outMPS = [np.ones((chi,chi,d))]*LL
    outMPS[0] = np.ones((1,chi,d))
    outMPS[-1] = np.ones((chi,1,d))
   

    return outMPS



def SVD_trunc(M: np.ndarray, epsTrunc: float, chiMax: int) -> tuple[np.ndarray,np.ndarray,np.ndarray, int]:
    """ Performs SVD and truncates at a given epsTrunc / chiMax """

    u, s, Vdag = rsvd(M,full_matrices=False)  

    Strunc = [sv for sv in s[:chiMax] if sv > epsTrunc]
    
    s = np.array(Strunc)
    sizeTruncS = np.size(s)


    # If we truncated the SVs, we should truncate accordingly 
    # the cols of U
    # and the rows of Vdag

    u = u[:,:sizeTruncS]
    Vdag = Vdag[:sizeTruncS,:]

    return u, s, Vdag, sizeTruncS






class myMPS:

    """ My own MPS implementation - Contains the following elements:
    LL(length),
    DD (physical dim), 
    chis(bond dims) [LL + 1 long, first and last are trivial bonds], 
  
    MPS (the matrices)
    Alist (the left-canonical matrices)
    Blist (the right-canonical matrices)
    Clist (the gammas for mixed canonical)
    
    SV (singular values (arrays)) [also LL+1]
    SVinv(inverse of SVs),
    form (can be 'L', 'R', 'C' or 'x')

    After bringing to some canonical form, the indices for the various sites (A) and bonds (S) should be 

           |         |         |                       |
    .S0--[A0]--S1--[A1]--S2--[A2]--.....--S(LL-1)--[A(LL-1)]--S[LL].

        that is, we have  *LL sites*  going from 0 to LL-1  
        and  *LL+1 bonds*  from 0 to LL, of which the 0th and LLth are trivial 
    """

    # Fancy stuff for performance saving
    __slots__ = ['LL','DD','MPS','chis','SV','SVinv','idx','normalized','canon','curr_form']




    def __init__(self, inputMPS: list[np.ndarray] = randMPS(LL=7, chi=20, d=2)):
      
        LL = len(inputMPS)
     
        idx = {
             'vL' : 0,
             'ph' : 2,
             'vR' : 1
        }

        self.idx = idx
        
        #Physical dimension - we assume it to be constant for the whole MPS
        DD = np.shape(inputMPS[1])[idx['ph']]  # not the most elegant way to extract it but eh..

        self.MPS: list[np.ndarray] = inputMPS  

        # Build the bond dimension list
        mChi = [ np.shape(mm)[idx['vL']] for mm in inputMPS ]
        mChi.append(np.shape(inputMPS[-1])[idx['vR']])
      
        mSV = [np.array(1.)] * (LL+1) # Empty for now 

        logging.info(f"MPS with length {LL} and physical d={DD}")
        logging.info(f"chi {mChi}")


        self.LL = LL  
        self.DD = DD  
  
        self.chis = mChi
        self.SV = mSV  
        self.SVinv = mSV  
        self.curr_form = 'x'  # By default we're not in any particular form 
        self.canon = False # by default not canon unless we say it is 
        self.normalized = False




    """ 
    The fat part: bringing an MPS to canonical form 
    """    


    def bringCan(self, epsTrunc: float=1e-12, epsNorm: float=1e-12, chiMax: int = 40) -> str:

        """ Brings input myMPS object to canonical form, returns ('form', 'lastSweep').

        Input: 
        - epsTrunc: below this epsilon we drop the SVs 
        - epsNorm: we decide that the state is normalized if |1-norm| < epsNorm
        - chiMax: max bond dimension we truncate to 
        
        We 
        1. perform a left QR sweep (and drop final piece so the norm should be 1)
        2. perform a right SVD sweep after the left one, and truncate the SV below epsTrunc
        3. check that we're still normalized after truncating and if not 
        4. perform another left SVD sweep 
        5. build the Gamma-Lambda-Gamma-Lambda canonical form

        According to the specified form, we identify the MPS matrices
        with those in Alist, Blist or Glist

        But in principle we should always be able to access the Alist,Blist,Glist directly from outside.
        """

        """ Try to uniform indices so that we're consistent with MPO.
        Internally, it's probably more efficient to work with the (vL, phys, vR) ordering,
        but outside we probably prefer (vL, vR, phys) to be consistent with the MPO conventions.
        
        So one way to do it could be to enforce as input/output the (vL,vR,ph) convention, 
        then convert internally at the beginning/end to/from the (vL,ph,vR) convention
        """

        

        LL = self.LL
        DD = self.DD

        chiIn = self.chis


        """ 
        # INDEX RELABELING: from (vL,vR,ph) to (vL,ph,vR)
        """ 
        
        MPS = [m.transpose(0,2,1) for m in self.MPS]

       
        ###############################################
        ###############################################
        ######### LEFT SWEEP       ####################
        ###############################################
        ###############################################
        

        _Alist = [np.array(1.)]*LL  # This will hold our A matrices 
        _Blist = [np.array(1.)]*LL  # This will hold our B matrices

        chiA = [1]*(LL+1)
        chiB = [1]*(LL+1)
    
        _Slist: list = [np.array([1.])]*(LL+1)
        


        # For the first sweep we do a QR decomp

        r = np.array(1.).reshape(1,1)
        for jj in range(0,LL):
           
            Mtilde = ncon([r, MPS[jj]], [[-1,1],[1,-2,-3]])  

            Mtr = np.reshape(Mtilde, (chiA[jj]*DD, chiIn[jj+1]))

            q,r  = qr(Mtr)  
            
            chiA[jj+1] = np.shape(q)[1]

            _Alist[jj] = np.reshape(q,(chiA[jj],DD,chiA[jj+1]))

    
        ###############################################
        ###############################################
        ######### RIGHT SWEEP   + TRUNCATION  #########
        ###############################################
        ###############################################

        # We will truncate here as well if given a epsTrunc/chiMax at input

        logging.info("Performing a right sweep")

        # For the first site only
        u = np.array(1.).reshape(1,1)
        s = np.array(1.).reshape(1)
        
        for jj in range(LL-1,0,-1):
        
            Mtilde = ncon([_Alist[jj], u, np.diag(s)], [[-1,-2,1],[1,2],[2,-3]])  

            Mtr = np.reshape(Mtilde, (chiA[jj],chiB[jj+1]*DD ))

            u, s, Vdag, sizeTruncS = SVD_trunc(Mtr, epsTrunc, chiMax)

            chiB[jj] = sizeTruncS
            _Slist[jj] = s
            _Blist[jj] = np.reshape(Vdag,(chiB[jj],DD,chiB[jj+1]))


        # Now the last site ("0")

        Mtilde = ncon([_Alist[0], u, np.diag(s)], [[-3,-1,1],[1,2],[2,-2]]) 

        # We should still reshape here!
        Mtr = np.reshape(Mtilde, (chiB[0],chiB[1]*DD))

        u, s, Vdag = rsvd(Mtr,full_matrices=False)  
        
        _Blist[0] = np.reshape(Vdag,(chiB[0],DD,chiB[1]))

        # The last factor should give the normalization

        tail = u @ s

        normsq = rc(tail*np.conjugate(tail))

        self.chis = chiB
        self.SV = _Slist

        logging.info(f"after R sweep chis: {chiB}")
        
        
        curr_form = 'R'

     
        # """
        #  Redo left sweep if not normalized after truncation 
        # """

        # FIXME: hack, always do 3rd sweep 
        self.normalized = 0
        
        if self.normalized == 0:
            logging.info("State not normalized after R sweep: Performing a Left Sweep again")
       
            Vdag = np.array(1.).reshape(1,1)
            s = np.array(1.).reshape(1)
            
            for jj in range(0,LL):
        
                Mtilde = ncon([np.diag(s), Vdag, _Blist[jj]], [[-1,1],[1,2],[2,-2,-3]])  
                Mtr = np.reshape(Mtilde, (chiA[jj]*DD, chiB[jj+1]))
                u, s, Vdag = rsvd(Mtr,full_matrices=False)  
                chiA[jj+1] = np.size(s)
                _Alist[jj] = np.reshape(u,(chiA[jj],DD,chiA[jj+1]))
                _Slist[jj+1] = s
        
            tail = s @ Vdag 
        
            normsq = rc(tail*np.conjugate(tail))



            logging.info(f"after 3nd sweep chis: {chiA}")

            logging.info(f"Final Norm = {normsq}")
            if abs(1.-normsq < epsTrunc) and self.checkSVsAreOne(epsTrunc): 
                self.normalized = 1
            else:
                logging.warning(f"Warning: state is not normalized even after 3 sweeps, |1-norm| = {abs(1.-normsq)} ")
                    
        
            curr_form = 'L'
            #print("3 sweeps")

            if( chiA != chiB): 
                raise ValueError("Something strange: after 3rd sweep chi's still changed")


        # Inverting SVs 

        _Sinvlist = _Slist[:]

        # here we invert SVs so be mindful 
        for ii, sli in enumerate(_Slist):
            _Sinvlist[ii] = np.array([ss**(-1) for ss in sli])
            if np.isnan(_Sinvlist[ii]).any():
                raise ZeroDivisionError("NaN when inverting SV's !!!! ")
        self.SVinv = _Sinvlist
        


        self.canon = True
        self.curr_form = curr_form 

        # Revert indices to outside convention
        if curr_form == 'L':
            self.MPS = [m.transpose(0,2,1) for m in _Alist]
        elif curr_form == 'R':
            self.MPS = [m.transpose(0,2,1) for m in _Blist]
            


        return  curr_form






    def set_form(self, mode: str = 'R') -> str:


        """ According to the mode selected, 
        set the MPS matrices to either the A,B or Gammas
        - mode: 'L' , 'R' or 'C' for left-can, right-can or mixed (gamma-lambda) canon form

        """
        if not self.normalized:
            self.bringCan()
            
        if not self.canon:
            #print("MPS not canonical, bringing to canon form")
            self.bringCan()
    

        if self.curr_form == mode:
            # no need to do anything, already in the mode we want 
            return self.curr_form

            
        Glist = [np.array(1.)]*self.LL

        if self.curr_form == 'R':
            # Building the canonical form from the B's 
            # And rebuilding the B's from the Gammas 

            Blist = self.MPS
            Alist = [np.array(1.)]*self.LL

            for jj in range(0,self.LL):
                Glist[jj] = ncon( [Blist[jj], np.diag(self.SVinv[jj+1])],[[-1,1,-3],[1,-2]])
                Alist[jj] = ncon([ np.diag(self.SV[jj]), Glist[jj] ], [[-1,1],[1,-2,-3]])



        elif self.curr_form == 'L':
            # Building the canonical form from the A's 
            # And rebuilding the B's from the Gammas 

            Alist = self.MPS
            Blist = [np.array(1.)]*self.LL

    
            for jj in range(0,self.LL):
                Glist[jj] = ncon( [np.diag(self.SVinv[jj]), Alist[jj]],[[-1,1],[1,-2,-3]])
                Blist[jj] = ncon([ Glist[jj] , np.diag(self.SV[jj+1])], [[-1,1,-3], [1,-2]])

        else: 
            raise ValueError("Wrong lastSweep")



        if mode == 'L':
            self.MPS = Alist
            self.curr_form = 'L'
            logging.info("Setting MPS matrices to LEFT form ")
            
        elif mode == 'R':
            self.MPS = Blist
            self.curr_form = 'R'
            logging.info("Setting MPS matrices to RIGHT form ")
            
        elif mode == 'C':
            self.MPS = Glist
            self.curr_form = 'C'
            logging.info("Setting MPS matrices to CANONICAL form ")
            
        else:
            logging.error("Wrong form specified, leaving undetermined ")
            self.curr_form = 'x'
        
        return self.curr_form






    def getNormSlow(self) -> complex:
        """ Calculates the MPS norm by ncon-ing everything"""
        
        MPSconj = [np.conj(m) for m in self.MPS]

        indicesM = []
        indicesMc = []
        for jj,m in enumerate(self.MPS, start = 1):   # we start the idx from 1 so ncon doesn't complain
            offsetM = (self.LL)*7
            offsetMc = (self.LL)*9

            indicesM.append([offsetM+jj,offsetM+jj+1,jj])
            indicesMc.append([offsetMc+jj,offsetMc+jj+1,jj])

        # Equate first and last indices  [we swap vL-vR if we're closing the TN..]
        indicesMc[0][0] = indicesM[0][0] 
        indicesMc[-1][1] = indicesM[-1][1]

        toContr = []
        [toContr.append(m) for m in self.MPS]
        [toContr.append(m) for m in MPSconj]

        idxList = indicesM[:]
        idxList.extend(indicesMc)

        # norm = ncon([self.MPS, MPSconj], [indicesM,indicesMc])
        normsq = rc(ncon(toContr,idxList))
        

        return np.sqrt(normsq)







    def getEntropies(self, checkCan: bool = True) -> list[float]:
        # Puts in canonical form if necessary and extracts the entropies 
        if checkCan and not self.canon:  
            numSVs = np.max(self.chis)
            logging.warning(f"Putting in canonical form and truncating at {numSVs}")
            self.bringCan(chiMax = numSVs, epsTrunc=1e-14)
        
        ents = []
        for lambdas in self.SV:
            ents.append(sum([-lam**2 *np.log(lam**2) for lam in lambdas]))
        
        return ents




    def overlap(self, withMPS: myMPS, conjugate: bool = False ) -> complex:
        
        return voverlap(self, withMPS, conjugate)


        

    def getNorm(self) -> float:
        norm = voverlap(self, self, conjugate=True)
        if np.imag(norm)/np.real(norm) < 1e-14: 
            norm = np.sqrt(np.real(norm))
        else:
            raise ArithmeticError("complex norm !?")
        return norm



    def checkNormalized(self, eps=1e-12) -> bool:
        if abs(1.-self.getNorm()) > eps:
            print(f"state is not normalized, norm = {self.getNorm()}")
            self.normalized = False
            return False
        else:
            self.normalized = True
            return True
        
    def checkSVsAreOne(self, eps=1e-12) -> bool:

        # The sum of squared SVDs should be 1
        deltas = [abs(1.-np.sum(sss**2)) for sss in self.SV]
        deltasAreLarge = [d > eps for d in deltas]
        if any(deltasAreLarge): 
            logging.warning(f"SVDs don't look too normalized, deltaMax = {np.max(deltas)} ")
            #logging.warning(deltas)
            self.normalized = False
            return False
        else:
            return True


    
    def checkCan(self, eps=1e-12) -> bool:
        """ We assume that having <psi|psi> = 1  _and_  sum SV^2 = 1 
            is good enough to say that it's in a canonical form """
            
        if self.checkNormalized() and self.checkSVsAreOne():
            self.canon = True
            return True
        else: 
            self.canon = False
            return False



    def expValOneSite(self, oper: np.ndarray, site: int) -> np.complex128:

        if(self.curr_form != 'R'):
            self.set_form(mode='R')

        # we can simply use the canonical form to compute it instantaneously

        conTen = [np.diag(self.SV[site]),np.diag(self.SV[site]),self.MPS[site],np.conj(self.MPS[site]),oper]
        conIdx = [[1,2],[1,3],[3,5,4],[2,5,6],[4,6]]

        return rc(ncon(conTen,conIdx))




def voverlap(bra: myMPS, ket: myMPS, conjugate: bool = False ) -> np.complex128:

    """Computes the overap of an MPS (bra) with another MPS (ket)
    Warning!!: This does only conjugate if you pass the "conjugate = True" argument 
    
     The order of the contraction goes like 

     2-4-6-8
     | | | |
     1-3-5-7-9... etc
        
    """

    if bra.LL != ket.LL :
        raise ValueError("Error: the lengths of the two MPSes do not coincide")
    if bra.DD != ket.DD:
        raise ValueError("Error: the physical dims of the two MPSes do not coincide")

    if conjugate: 
        braMPS = [np.conj(m) for m in bra.MPS]
    else:
        braMPS = bra.MPS

    ketMPS = ket.MPS

    # Start from left 
    blobL = ncon([ketMPS[0], braMPS[0]] ,[[1,-1,2],[1,-2,2]])

    #Now the bulk 
    for jj in range(1,ket.LL):
        logging.debug(f"{jj}: shapes: {np.shape(blobL)}, {np.shape(ketMPS[jj])}")

        blobL = ncon( [blobL, ketMPS[jj]], [[1,-3],[1,-1,-2]])
        blobL = ncon( [blobL, braMPS[jj]],[[-1,1,2],[2,-2,1]])

        logging.debug(f"{jj}: after: {np.shape(blobL)}")
        #print(blobL)

    # Close the last site         
    overl = ncon( [blobL], [[1,1]] )    
    #print(f"{overl=}, {type(overl)=}")

    return rc(overl)

