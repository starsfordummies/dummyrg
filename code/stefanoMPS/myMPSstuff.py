# Last modified: 2022/07/12 17:36:55

import numpy as np
from numpy import linalg as LA
from tensornetwork import ncon
import logging

# TODO: should I switch to Luca's convention for MPS indices, 
# v_L , v_R , phys 
# for the peace of mind? 


def randMPS(LL: int=10, chi: int=5, d: int=2):

    """ MPS Builder function - OBC for now 
    Returns a list of length LL of random tensors with bond dimension chi 
    and physical dimension d 
    """

    logging.info(f"Building random MPS with length {LL}, chi = {chi} and physical d={d}")

    # Build the first element of the MPS A_1 (actually 0th elem of the list)
    myMPS = [ np.random.rand(1,d,chi) + 1j*np.random.rand(1,d,chi) ]

    #Build all the others, A_2 to A_(N-1)
    for ii in range(2,LL):  # 2 to N-1, so actually building N-2 elements
         myMPS.append(np.random.rand(chi,d,chi) + 1j*np.random.rand(chi,d,chi) )
 
    # Build the Nth element (actually element N-1 of the list)
    myMPS.append( np.random.rand(chi,d,1) + 1j*np.random.rand(chi,d,1))

    return myMPS




class myMPS:

    """ My own MPS implementation - Contains the following elements:
    LL(length),
    DD (physical dim), 
    chis(bond dims) [LL + 1 long, first and last are trivial bonds], 
  
    indices (for contraction),
    offIndices is the offset factor which we use to label the virtual indices to be contracted, 
    we take it to be >> 1 so we don't get mixed indices labels 
    
    MPS (the matrices)
    Alist (the left-canonical matrices)
    Blist (the right-canonical matrices)
    Clist (the gammas for mixed canonical)
    
    SV (singular values (arrays)) [also LL+1]
    SVinv(inverse of SVs),
    form (can be 'L', 'R', 'C' or 'x')

    
    """


    def __init__(self, inputMPS: list=randMPS(LL=7, chi=20, d=2), offIndices: int=5, can: bool=True):
      
        LL = len(inputMPS)

        #Physical dimension - we assume it to be constant for the whole MPS
        DD = np.shape(inputMPS[1])[1]  # not the most elegant way to extract it but eh..

        self.MPS = inputMPS  

        mChi = [ np.shape(mm)[0] for mm in inputMPS ]
        mChi.append(np.shape(inputMPS[-1])[-1])
        #inputMPS[1:] ]  # Should be LL-1 long
        mSV = [1] * (LL+1) # Empty for now 

        print(f"MPS with length {LL} and physical d={DD}")
        print(f"chi {mChi}")


        # We could build indices like: 
        # physical go from 1 to L   (ncon doesn't like 0 as idx..)
        # virtual from L+1 up to 2L 
        """
        # Should we use a DICT for the indices? 
        indices = [{ 
            'vL': 0,
            'ph': 1,
            'vR': offIndices*LL+1
            }]

        """

        indices=[]
        # Try to build the indices in a symmetrical way, 
        # without treating differently the 1st and last site
        # (anyway should be a trivial contraction if they're just dim 1 scalars on that leg)
        for midx in range(1,LL+1):  #Builds LL sets of indices from 1 to LL
            indices.append({ 'vL': offIndices*LL+midx, 'ph': midx, 'vR': offIndices*LL+midx+1 })

        self.LL = LL  
        self.DD = DD  

        self.indices = indices
  
        self.chis = mChi
        self.SV = mSV  
        self.SVinv = mSV  
        self.form = 'x'  # By default we're not in any particular form 

        ## TODO: maybe I should put a call to bringCanonical here
        # TODO: we should allow to pass pass cut parameters here 
        #if bringcanonical: self.bringCan()


        
       
        

    # TODO: if I understand correctly, TenPy does QR for the left sweep and SVD for 
    # TODO: the right one, I think it's because it's faster.
    # TODO: should I worry about implementing it?
        
    def bringCan(self, mode: str='R', epsTrunc: float=1e-10, epsNorm: float=1e-12, chiMax: int = 40):

        """ Brings input myMPS object to canonical form, returns 'form'.
        We 
        1. perform a left SVD sweep (and drop final piece so the norm should be 1)
        2. perform a right SVD sweep after the left one, and truncate the SV below epsTrunc
        3. check that we're still normalized after truncating and if not 
        4. perform another left SVD sweep [TODO: as of now we're doing it anyway]
        5. build the Gamma-Lambda-Gamma-Lambda canonical form

        According to the specified form, we identify the MPS matrices
        with those in Alist, Blist or Glist

        But in principle we should always be able to access the Alist,Blist,Glist directly from outside.
        """

        LL = self.LL
        DD = self.DD

        chiIn = self.chis

        MPS = self.MPS

       


    
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
        We work with 
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

        U, S, Vdag = LA.svd(Mtr,full_matrices=0)

        logging.info("First SVD:")
        logging.info(f"{np.shape(Alist[LL-1])} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
        logging.info(f"SV = {S}")
        logging.info(f"chi_{LL} (nonzero SVs) = {np.size(S)}")

        # Truncating the SVs at epsTrunc/chiMax
        Strunc = []
        for (idxs, sv) in enumerate(S):
            if sv > epsTrunc:
                if idxs >= chiMax:
                    logging.warning(f"Truncating @ {chiMax}, latest SV = {sv}")
                    break
                else:
                    Strunc.append(sv)
            else:
                # We could implement a break here as well, since they should be descending
                break
        
        S = np.array(Strunc)
        sizeTruncS = np.size(S)

        logging.info(f"chi_{LL} (truncated SVs) = {sizeTruncS}")

        # If we truncated the SVs, we should truncate accordingly the cols of U
        # and the rows of Vdag

        U = U[:,:sizeTruncS]
        Vdag = Vdag[:sizeTruncS,:]

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

            U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
            
            logging.debug( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
            logging.info(f"chi_{pjj} (nonzero SVs) = {np.size(S)}")


            #Truncate here as well 
            Strunc = []
            for (idxs, sv) in enumerate(S):
                if sv > epsTrunc:
                    if idxs >= chiMax:
                        logging.warning(f"Truncating @ {chiMax}, latest SV = {sv}")
                        break
                    else:
                        Strunc.append(sv)
                else:
                    # We could implement a break here as well, since they should be descending
                    break
                          
            S = np.array(Strunc) 
            sizeTruncS = np.size(S)
            

            logging.info(f"chi_{pjj} (truncated SVs) = {sizeTruncS}")

            # If we truncated the SVs, we should truncate accordingly the cols of U
            # and the rows of Vdag

            U = U[:,:sizeTruncS]
            Vdag = Vdag[:sizeTruncS,:]


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

        if np.abs( normsq - 1.) < epsNorm:
            self.normalized = 1 
        else:
            self.normalized = 0 
            #logging.warning(f"MPS not normalized, normsq = {normsq} (|1-norm| = {np.abs( normsq - 1.)})")
        
            #TODO: here we can either divide the SVDs by the norm to re-normalize,
            #TODO: or if we want to do better we redo a left sweep
            
        
        logging.info(f"SVs = {Slist}")

        self.chis = chiB
        self.SV = Slist

        #self.form = 'R'

        logging.info(f"after R sweep chis: {chiB}")
        logging.info("and the SVs squared:")
        
        # The sum of squared SVDs should be 1, right? 
        #deltas=  [1.-np.sum(sss**2) for sss in Slist]
        #print(deltas)




        #TODO: implement check if normalized etc..
        #############################################
        ####### REDO LEFT SWEEP ########
        ###############################################
        
        logging.info("Performing a Left Sweep again")
        
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
        #print("delta (1-SVs^2):")
        deltas=  [1.-np.sum(sss**2) for sss in Slist]
        deltasAreBig = [d > epsTrunc for d in deltas]
        if any(deltasAreBig): 
            logging.warning("SVDs don't look too normalized, deltas = ")
            logging.warning(deltas)

        # The sum of squared SVDs should be 1
             
        logging.info(f"Final Norm = {normsq}")
        if abs(1.-normsq < epsTrunc): 
            self.normalized = 1
        else:
            logging.error(f"Warning: state is not normalized even after 3rd sweep, |1-norm| = {abs(1.-normsq)} ")
                
       



        ###############################################
        ###############################################
        #######    CANONICAL FORM    ##################
        ###############################################
        ###############################################



        logging.info("Building Gamma-Lambda canonical form")

        Sinvlist = Slist[:]

        # FIXME: here we invert SVs so be mindful 
        logging.debug(f"Types:  {type(Slist)}, {type(Sinvlist)},{type(Slist[2])},{type(Sinvlist[2])}")
        for ii in range(0,len(Slist)):
            Sinvlist[ii] = [ss**(-1) for ss in Slist[ii]]
            if np.isnan(Sinvlist[ii]).any():
                logging.error("NaN when inverting SV's !!!! ")
                exit(-1)
        self.SVinv = Sinvlist
        

        Glist = [1]*LL

        # Building the canonical form from the B's 
        #Glist[0] = Blist[0]
        #for jj in range(0,LL):
        #    logging.debug(f"Gam[{jj}],{np.shape(Sinvlist[jj-1])},{np.shape(Blist[jj])}")
        #    Glist[jj] = ncon( [Blist[jj], np.diag(Sinvlist[jj+1])],[[-1,-2,1],[1,-3]])

        # Building the canonical form from the A's 
        # And rebuilding the B's from the Gammas 

        #Glist[0] = Alist[0]
        for jj in range(0,LL):
            logging.debug(f"Gam[{jj}],{np.shape(Sinvlist[jj])},{np.shape(Alist[jj])}")
            Glist[jj] = ncon( [np.diag(Sinvlist[jj]), Alist[jj]],[[-1,1],[1,-2,-3]])
            
            Blist[jj] = ncon([ Glist[jj] , np.diag(Slist[jj+1])], [[-1,-2,1], [1,-3]])



        """ According to the mode selected, 
        set the MPS matrices to either the A,B or Gammas
        """

        if mode == 'L':
            self.MPS = Alist
            self.form = 'L'
            self.chis = chiA
            logging.info("Setting MPS matrices to LEFT form ")
        elif mode == 'R' or mode == 'LR': #for backwards compatibility
            self.MPS = Blist
            self.form = 'R'
            self.chis = chiB
            logging.info("Setting MPS matrices to RIGHT form ")
        elif mode == 'C':
            self.MPS = Glist
            self.form = 'C'
            self.chis = chiA
            logging.info("Setting MPS matrices to CANONICAL form ")
        else:
            logging.error("Wrong form specified, leaving undetermined ")
            self.form = 'x'
        

        return self.form







    def updatePhysIndices( self, openIndices : list ):
        """ Updates the labels of the physical indices using an input list """
        if len(openIndices) != (self.LL):
            logging.warning("Wrong length of open indices list, doing nothing")
            pass
        else:
            indices = self.indices
            for (ii, idx) in enumerate(indices):
                idx["ph"] = openIndices[ii]
            #self.indices = indices  # Is this even necessary ? 




    def getPhysIndices( self ):
        """ Returns a list with the labels of the physical indices """
       
        indices = self.indices
        listPhys = []
        [listPhys.append(idx["ph"]) for idx in indices]

        return listPhys
              


              
    
    def updateEdgeIndices( self, openIndices : list ):
        """ Updates the labels of the edge indices using an input list """
        if len(openIndices) != 2:
            logging.warning("Wrong length of edge indices list, doing nothing")
            pass
        else:
            indices = self.indices
        
            indices[0]["vL"] = openIndices[0]
            indices[-1]["vR"] = openIndices[1]
            #self.indices = indices  # Is this even necessary ?
    


    
    def getEdgeIndices( self ):
        """ Returns a list with the labels of the edge indices """
       
        indices = self.indices
        listEdge = [ indices[0]["vL"], indices[-1]["vR"] ]

        return listEdge
         



    def getIndices(self):
        """ Returns a list of lists with the indices for contracting the TN with ncon """
        iL = []
        for el in self.indices:

            # Build the (2 or 3-valued) indices list
            idxs = []
            
            if el["vL"] != 0:
                idxs.append(el["vL"])

            idxs.append(el["ph"])

            if el["vR"] != 0:
                idxs.append(el["vR"])
            
            iL.append(idxs)
        
        return iL





    def getNorm(self):
        """ Calculates the MPS norm by ncon-ing everything"""
        
        MPSconj = [np.conj(m) for m in self.MPS]

        indicesM = []
        indicesMc = []
        for jj,m in enumerate(self.MPS,1):
            offsetM = (self.LL)*7
            offsetMc = (self.LL)*9

            indicesM.append([offsetM+jj,jj,offsetM+jj+1])
            indicesMc.append([offsetMc+jj,jj,offsetMc+jj+1])

        # Equate first and last indices
        indicesMc[0][0] = indicesM[0][0]
        indicesMc[-1][-1] = indicesM[-1][-1]

        toContr = []
        [toContr.append(m) for m in self.MPS]
        [toContr.append(m) for m in MPSconj]

        idxList = indicesM[:]
        idxList.extend(indicesMc)

        # norm = ncon([self.MPS, MPSconj], [indicesM,indicesMc])
        norm = np.real_if_close(ncon(toContr,idxList))

        return norm








    def getEntropies(self):
        # Puts in canonical form if necessary and extracts the entropies 
        if(self.form != 'cccc'):  # TODO: always ops 
            logging.info("Putting in Right canonical form")
            self.bringCan()
            #print(f"isnormalized? {self.normalized}")
        
        #TODO: check if the formula is correct, factors sqrts etc
        ents = []
        for lambdas in self.SV:
            si = 0.
            si = sum([-lam*np.log(lam) for lam in lambdas])
            ents.append(si)
        
        return ents


# TODO: refactor this (probably we don't even need it )
def expValOneSite(iMPS: object, oper: np.array, site: int):

    if(iMPS.form != 'LR'):
        iMPS.bringCan(mode='LR',epsTrunc=1e-12)

    conTen = [np.diag(iMPS.SV[1]),np.diag(iMPS.SV[1]),iMPS.MPS[2],np.conj(iMPS.MPS[2]),oper]
    conIdx = [[1,2],[1,3],[3,4,5],[2,6,5],[4,6]]

    return np.real_if_close(ncon(conTen,conIdx))



