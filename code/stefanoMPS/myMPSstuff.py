# Last modified: 2022/08/05 17:28:12

from __future__ import annotations

import numpy as np
from numpy import linalg as LA
from tensornetwork import ncon
import logging

# MPS Indices convention:
# v_L , v_R , phys 


def randMPS(LL: int=10, chi: int=5, d: int=2) -> list[np.array]:

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


def plusState(LL: int=10) -> list[np.array]:

    """ Returns a product "plus" state
    """

    chi =1 
    d = 2
    logging.info(f"Building product |+> state of length {LL} (physical d={d})")

    # Build the first element of the MPS A_1 (actually 0th elem of the list)
    plus = [1./np.sqrt(d)]*d
    outMPS = [np.array(plus).reshape(chi,chi,d)]*LL

    return outMPS

def bigEntState(LL: int=10) -> list[np.array]:

    """ Returns a product "plus" state
    """

    chi = 2
    d = 2
    logging.info(f"Building product |+> state of length {LL} (physical d={d})")

    outMPS = [np.ones((chi,chi,d))]*LL
    outMPS[0] = np.ones((1,chi,d))
    outMPS[-1] = np.ones((chi,1,d))
   

    return outMPS



# TODO: maybe unnecessary now 
def truncSVs(S: np.array, epsTrunc: float, chiMax: int) -> list[float]:
    # Truncating the SVs at epsTrunc/chiMax
    # We assume that the input S is sorted in descending order (should be the case for SVD output)
    Strunc = [sv for sv in S[:chiMax] if sv > epsTrunc]

    return np.array(Strunc)



def SVD_trunc(M: np.array, epsTrunc: float, chiMax: int) -> tuple[np.array,np.array,np.array, int]:
    """ Performs SVD and truncates at a given epsTrunc / chiMax """

    U, S, Vdag = LA.svd(M,full_matrices=0)  

    Strunc = [sv for sv in S[:chiMax] if sv > epsTrunc]

    S = np.array(Strunc)
    sizeTruncS = np.size(S)


    # If we truncated the SVs, we should truncate accordingly 
    # the cols of U
    # and the rows of Vdag

    U = U[:,:sizeTruncS]
    Vdag = Vdag[:sizeTruncS,:]

    return U, S, Vdag, sizeTruncS






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

    def __init__(self, inputMPS: list=randMPS(LL=7, chi=20, d=2)):
      
        LL = len(inputMPS)
     
        idx = {
             'vL' : 0,
             'ph' : 2,
             'vR' : 1
        }

        self.idx = idx
        
        #Physical dimension - we assume it to be constant for the whole MPS
        DD = np.shape(inputMPS[1])[idx['ph']]  # not the most elegant way to extract it but eh..

        self.MPS = inputMPS  

        mChi = [ np.shape(mm)[idx['vL']] for mm in inputMPS ]
        mChi.append(np.shape(inputMPS[-1])[idx['vR']])
      
        mSV = [1] * (LL+1) # Empty for now 

        logging.info(f"MPS with length {LL} and physical d={DD}")
        logging.info(f"chi {mChi}")


        self.LL = LL  
        self.DD = DD  
  
        self.chis = mChi
        self.SV = mSV  
        self.SVinv = mSV  
        self.curr_form = 'x'  # By default we're not in any particular form 
        self.canon = False # by default not canon unless we say it is 




    """ 
    The fat part: bringing an MPS to canonical form 
    """    


    # TODO: use QR for the first sweep
        
    def bringCan(self, epsTrunc: float=1e-12, epsNorm: float=1e-12, chiMax: int = 40) -> tuple[str, str]:

        """ Brings input myMPS object to canonical form, returns ('form', 'lastSweep').

        Input: 
        - epsTrunc: below this epsilon we drop the SVs 
        - epsNorm: we decide that the state is normalized if |1-norm| < epsNorm
        - chiMax: max bond dimension we truncate to 
        
        We 
        1. perform a left SVD sweep (and drop final piece so the norm should be 1)
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
        
        logging.info("Performing a Left Sweep")
        

        chiA = [1]*(LL+1)
        Slist = [np.array([1.])]*(LL+1)
        Alist = [1.]*LL  # This will hold our A matrices for the LeftCanonical form

        # First site:

        Mtilde = MPS[0]
        Mtr = np.reshape(Mtilde, (chiA[0]*DD, chiIn[1]))

        U, S, Vdag = LA.svd(Mtr,full_matrices=0)

        chiA[1] = np.size(S)
        Slist[1] = S  

        U = np.reshape(U,(chiA[0],DD,chiA[1]))


        logging.info("First SVD:")
        logging.info(f"{np.shape(MPS[0])} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
        logging.info(f"SV = {S}")
        logging.info(f"chi_1 (nonzero SVs) = {np.size(S)}")


        Alist[0] = U

        """ 
        Going to the next one, we need to make some products and some reshaping.
        """

        for jj in range(1,LL-1):
            pjj = jj+1  # The labels we're using start from 1, so for printing use this index
            logging.info(f"Building leftcanon form for M[{pjj}]. First build Mtilde[{pjj}] = SV.Vdag.M[{pjj}]")

            Mtilde = ncon([np.diag(S), Vdag, MPS[jj]], [[-1,1],[1,2],[2,-2,-3]])  

            logging.info(f"Mtilde[{pjj}] = {np.shape(Mtilde)}  - Reshape it as chiA[{pjj}]*d, chiIn[{pjj}")
            Mtr = np.reshape(Mtilde, (chiA[jj]*DD, chiIn[jj+1]))

            U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
            
            logging.info( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
            logging.info(f"chi_{pjj+2} (nonzero SVs) = {np.size(S)}")

            chiA[jj+1] = np.size(S)

            logging.debug(f"Sum SVDs on left {jj}: {sum(S)}")
            
            # Reshape U
            U = np.reshape(U,(chiA[jj],DD,chiA[jj+1]))

            Slist[jj+1] = S 
            Alist[jj] = U


        # Now the last site ("LL-1")
        logging.info(f"Last site is just a vector, ie. tensor has shape {np.shape(MPS[LL-1])}")

        Mtilde = ncon([np.diag(S),Vdag,MPS[LL-1]], [[-1,1],[1,2],[2,-2,-3]]) 

        logging.info(f"Mtilde = {np.shape(Mtilde)}")

        # We should still reshape here!
        Mtr = np.reshape(Mtilde, (chiA[LL-1]*DD, chiIn[LL]))

        U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
        
        logging.info( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
        logging.info(f"chi_{LL} (nonzero SVs) = {np.size(S)}")

        # I guess we can reshape the last simply as

        logging.info(f"From {np.shape(U)} ")

        U = np.reshape(U,(chiA[LL-1], DD, chiA[LL]))
        logging.info(f"to {np.shape(U)} ")

        Alist[LL-1] = U

        # The last factor should just give the normalization
        #tail = S @ Vdag 

        logging.info(f"after L sweep chis: {chiA}")
       
    
        ###############################################
        ###############################################
        ######### RIGHT SWEEP   + TRUNCATION  #########
        ###############################################
        ###############################################

        # We will truncate here as well if given a epsTrunc/chiMax at input

        logging.info("Performing a right sweep")

        Blist = [1.]*LL  # This will hold our B matrices for the R sweep
        chiB = [1]*(LL+1)
    
        # First site:

        Mtilde = Alist[LL-1]
        Mtr = np.reshape(Mtilde, (chiA[LL-1],chiB[LL]*DD ))

        """ TODO: I implemented a separate func for all this: 
        U, S, Vdag = LA.svd(Mtr,full_matrices=0)

        logging.info("First SVD:")
        logging.info(f"{np.shape(Alist[LL-1])} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
        logging.info(f"SV = {S}")
        logging.info(f"chi_{LL} (nonzero SVs) = {np.size(S)}")  

        
        # Truncating the SVs at epsTrunc/chiMax
        S = truncSVs(S, epsTrunc, chiMax)
        sizeTruncS = np.size(S)

        logging.info(f"chi_{LL} (truncated SVs) = {sizeTruncS}")

        # If we truncated the SVs, we should truncate accordingly the cols of U
        # and the rows of Vdag

        U = U[:,:sizeTruncS]
        Vdag = Vdag[:sizeTruncS,:]
        """

        U, S, Vdag, sizeTruncS = SVD_trunc(Mtr, epsTrunc, chiMax)


        chiB[LL-1] = sizeTruncS
        Slist[LL-1] = S  
        logging.debug(f"reshape {np.shape(Vdag)} into {chiB[LL-1]}x{DD}x{chiB[LL]}")
        
        Vdag = np.reshape(Vdag,(chiB[LL-1],DD,chiB[LL]))

        Blist[LL-1] = Vdag

        """ 
        Going to the next one, we need to make some products and some reshaping.
        We work with 
        """

        for _jj in range(2,LL):
            idx = LL-_jj  # starts at LL-2 , ends at 1  
            pjj = idx+1  # The labels we're using start from 1, so for printing use this index
            logging.debug(f"Building rightcanon form for M[{pjj}]. First build Mtilde[{pjj}] = M[{pjj}.U.S]")

            Mtilde = ncon([Alist[idx], U, np.diag(S)], [[-1,-2,1],[1,2],[2,-3]])  

            logging.debug(f"Mtilde[{pjj}] = {np.shape(Mtilde)}  - Reshape it as chiIn_{pjj} , chiB_{pjj+1}*d")
            Mtr = np.reshape(Mtilde, (chiA[idx],chiB[idx+1]*DD ))

            
            """
            U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
            
            logging.debug( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
            logging.info(f"chi_{pjj} (nonzero SVs) = {np.size(S)}")


            #Truncate here as well 
            S = truncSVs(S, epsTrunc, chiMax)

            sizeTruncS = np.size(S)
            

            logging.info(f"chi_{pjj} (truncated SVs) = {sizeTruncS}")

            # If we truncated the SVs, we should truncate accordingly the cols of U
            # and the rows of Vdag

            U = U[:,:sizeTruncS]
            Vdag = Vdag[:sizeTruncS,:]
            """ 

            U, S, Vdag, sizeTruncS = SVD_trunc(Mtr, epsTrunc, chiMax)

            chiB[idx] = sizeTruncS
            
            Slist[idx] = S 

            # Reshape Vdag

            Vdag = np.reshape(Vdag,(chiB[idx],DD,chiB[idx+1]))

            Blist[idx] = Vdag


        # Now the last site ("L-1")
        logging.debug(f"Last site is just a vector, ie. tensor has shape {np.shape(MPS[0])}")

        Mtilde = ncon([Alist[0], U, np.diag(S)], [[-3,-1,1],[1,2],[2,-2]]) 

        logging.debug(f"Mtilde = {np.shape(Mtilde)}")

        # We should still reshape here!
        Mtr = np.reshape(Mtilde, (chiB[0],chiB[1]*DD))

        U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
        
        logging.debug( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
        logging.info(f"r[{LL}] (nonzero SVs) = {np.size(S)}")

        # we can reshape the last simply as

        logging.info(f"From {np.shape(Vdag)} ")
        Vdag = np.reshape(Vdag,(chiB[0],DD,chiB[1]))
        logging.info(f"to {np.shape(Vdag)} ")

        Blist[0] = Vdag

        # The last factor should give the normalization

        tail = U @ S

        normsq = np.real_if_close(tail*np.conjugate(tail))

        deltas=  [1.-np.sum(sss**2) for sss in Slist]
        deltasAreSmall = [d < epsTrunc for d in deltas]
   
        if np.abs( normsq - 1.) < epsNorm and any(deltasAreSmall):
            self.normalized = 1 
        else:
            self.normalized = 0 
   
            
        
        logging.info(f"SVs = {Slist}")

        self.chis = chiB
        self.SV = Slist

        logging.info(f"after R sweep chis: {chiB}")
        logging.info("and the SVs squared:")
        
        
        curr_form = 'R'


     
        """
         Redo left sweep if not normalized after truncation 
        """
        
        if self.normalized == 0:
            logging.info("State not normalized after R sweep: Performing a Left Sweep again")
            
            Mtilde = Blist[0]
            Mtr = np.reshape(Mtilde, (chiA[0]*DD, chiB[1]))

            U, S, Vdag = LA.svd(Mtr,full_matrices=0)

            chiA[1] = np.size(S)
            Slist[1] = S  

            U = np.reshape(U,(chiA[0],DD,chiA[1]))

            Alist[0] = U

            for jj in range(1,LL-1):
        
                Mtilde = ncon([np.diag(S), Vdag, Blist[jj]], [[-1,1],[1,2],[2,-2,-3]])  
                Mtr = np.reshape(Mtilde, (chiA[jj]*DD, chiB[jj+1]))
                U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
                chiA[jj+1] = np.size(S)
                U = np.reshape(U,(chiA[jj],DD,chiA[jj+1]))
                Slist[jj+1] = S 
                Alist[jj] = U

            Mtilde = ncon([np.diag(S),Vdag,Blist[LL-1]], [[-1,1],[1,2],[2,-2,-3]]) 
            Mtr = np.reshape(Mtilde, (chiA[LL-1]*DD, chiB[LL]))

            U, S, Vdag = LA.svd(Mtr,full_matrices=0)  

            U = np.reshape(U,(chiA[LL-1], DD, chiA[LL]))
        
            Alist[LL-1] = U
        
            tail = S @ Vdag 

            normsq = np.real_if_close(tail*np.conjugate(tail))


            logging.info(f"after 2ndL sweep chis: {chiA}")

            # The sum of squared SVDs should be 1
            #print("delta (1-SVs^2):")
            deltas = [1.-np.sum(sss**2) for sss in Slist]
            deltasAreSmall = [d < epsTrunc for d in deltas]
            if any(deltasAreSmall): 
                logging.info("SVDs don't look too normalized, deltas = ")
                logging.info(deltas)

            logging.info(f"Final Norm = {normsq}")
            if abs(1.-normsq < epsTrunc) and any(deltasAreSmall): 
                self.normalized = 1
            else:
                logging.warning(f"Warning: state is not normalized even after 3 sweeps, |1-norm| = {abs(1.-normsq)} ")
                    
        
            curr_form = 'L'

            if( chiA != chiB): 
                raise ValueError("Something strange: after 3rd sweep chi's still changed")


        ###############################################
        ###############################################
        #######    CANONICAL FORM    ##################
        ###############################################
        ###############################################



        logging.info("Building Gamma-Lambda canonical form")

        Sinvlist = Slist[:]

        # here we invert SVs so be mindful 
        logging.debug(f"Types:  {type(Slist)}, {type(Sinvlist)},{type(Slist[2])},{type(Sinvlist[2])}")
        for ii in range(0,len(Slist)):
            Sinvlist[ii] = [ss**(-1) for ss in Slist[ii]]
            if np.isnan(Sinvlist[ii]).any():
                raise ZeroDivisionError("NaN when inverting SV's !!!! ")
        self.SVinv = Sinvlist
        


        self.canon = True
        self.curr_form = curr_form 

        # Revert indices to outside convention
        if curr_form == 'L':
            self.MPS = [m.transpose(0,2,1) for m in Alist]
        elif curr_form == 'R':
            self.MPS = [m.transpose(0,2,1) for m in Blist]
            


        return  curr_form


    def set_form(self, mode: str = 'R'):


        """ According to the mode selected, 
        set the MPS matrices to either the A,B or Gammas
        - mode: 'L' , 'R' or 'C' for left-can, right-can or mixed (gamma-lambda) canon form

        """
        if not self.canon:
            print("MPS not canonical, bringing to canon form")
            self.bringCan()


        if self.curr_form == mode:
            # no need to do anything, already in the mode we want 
            return self.curr_form

            
        Glist = [1]*self.LL

        # back to working convention TODO: might not need to if we ncon right
        work = [m.transpose(0,2,1) for m in self.MPS]

        if self.curr_form == 'R':
            # Building the canonical form from the B's 
            # And rebuilding the B's from the Gammas 

            Blist = work
            Alist = Blist[:]

            Glist[0] = Blist[0]
            for jj in range(0,self.LL):
                Glist[jj] = ncon( [Blist[jj], np.diag(self.SVinv[jj+1])],[[-1,-2,1],[1,-3]])
                Alist[jj] = ncon([ np.diag(self.SV[jj]), Glist[jj] ], [[-1,1],[1,-2,-3]])

        elif self.curr_form == 'L':
            # Building the canonical form from the A's 
            # And rebuilding the B's from the Gammas 

            Alist = work
            Blist = Alist[:]

    
            for jj in range(0,self.LL):
                Glist[jj] = ncon( [np.diag(self.SVinv[jj]), Alist[jj]],[[-1,1],[1,-2,-3]])
                Blist[jj] = ncon([ Glist[jj] , np.diag(self.SV[jj+1])], [[-1,-2,1], [1,-3]])

        else: 
            raise ValueError("Wrong lastSweep")


        """ 
        Done. Now for output transpose back all matrices 
            from (vL,ph,vR) to (vL,vR,ph)  
        """


      


        if mode == 'L':
            Alist = [m.transpose(0,2,1) for m in Alist]
            self.MPS = Alist
            self.form = 'L'
            logging.info("Setting MPS matrices to LEFT form ")
        elif mode == 'R' or mode == 'LR': #for backwards compatibility
            Blist = [m.transpose(0,2,1) for m in Blist]
            self.MPS = Blist
            self.form = 'R'
            logging.info("Setting MPS matrices to RIGHT form ")
        elif mode == 'C':
            Glist = [m.transpose(0,2,1) for m in Glist]
            self.MPS = Glist
            self.form = 'C'
            logging.info("Setting MPS matrices to CANONICAL form ")
        else:
            logging.error("Wrong form specified, leaving undetermined ")
            self.form = 'x'
        
        return self.form




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
        norm = np.real_if_close(ncon(toContr,idxList))

        return norm







    # it's unlikely that we want to be truncating here.. 
    def getEntropies(self, numSVs: int = 0 ) -> list[float]:
        # Puts in canonical form if necessary and extracts the entropies 
        if not self.canon or not self.normalized:  
            if numSVs == 0: numSVs = np.max(self.chis)
            logging.warning(f"Putting in canonical form and truncating at {numSVs}")
            self.bringCan(chiMax = numSVs, epsTrunc=1e-14)

        
        #TODO: check if the formula is correct, factors sqrts etc
        ents = []
        for lambdas in self.SV:
            #si = sum([-lam*np.log(lam) for lam in lambdas])
            ents.append(sum([-lam**2 *np.log(lam**2) for lam in lambdas]))
        
        return ents




    def overlap(self, withMPS: myMPS, conjugate: bool = False ) -> complex:
        
        return voverlap(self, withMPS, conjugate)


        

    def getNorm(self) -> float:
        norm = voverlap(self, self, conjugate=True)
        if np.imag(norm)/np.real(norm) < 1e-15: norm = np.real(norm)
        return norm 



    def expValOneSite(self, oper: np.array, site: int) -> complex:

        if(self.form != 'R'):
            self.set_form(mode='R')

        # we can simply use the canonical form to compute it instantaneously

        conTen = [np.diag(self.SV[site]),np.diag(self.SV[site]),self.MPS[site],np.conj(self.MPS[site]),oper]
        conIdx = [[1,2],[1,3],[3,5,4],[2,5,6],[4,6]]

        return np.real_if_close(ncon(conTen,conIdx))




def voverlap(bra: myMPS, ket: myMPS, conjugate: bool = False ) -> complex:

    """Computes the overap of an MPS (bra) with another MPS (ket)
    Warning!!: This does only conjugate if you pass the "conjugate = True" argument 
    
     The order of the contraction goes like 

     2-4-6-8
     | | | |
     1-3-5-7-9... etc
        
    """

    if bra.LL != ket.LL :
        raise ValueError("Error: the sizes of the two MPSes do not coincide")
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

    return np.real_if_close(overl)

