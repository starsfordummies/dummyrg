import numpy as np
from numpy import linalg as LA
from tensornetwork import ncon
import logging

#import math 
#import copy 


# MPS Builder function - OBC for now 
def randMPS(LL: int, chi: int=5, d: int=2):

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
    chis(bond dims), 
    indices (for contraction),
    SV (singular values (arrays)),
    SVinv(inverse of SVs),
    form (can be 'L', 'R', 'C' or 'x')
    """

    def __init__(self, inputMPS: list=randMPS(7), offIndices: int=5):
      
        LL = len(inputMPS)

        DD = np.shape(inputMPS[1])[1]  # not the most elegant way to extract it but eh..

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

        #Build all the others, M_2 to M_(N-1)
        for midx in range(2,LL):  # 2 to N-1, so actually building N-2 elements
            indices.append({ 'vL': offIndices*LL+midx-1, 'ph': midx, 'vR': offIndices*LL+midx })

        # Build the Nth element (actually element N-1 of the list)
        indices.append({ 'vL': (offIndices+1)*LL - 1, 'ph': LL, 'vR': 0 })
        """

        indices=[]
        # Try to redo the indices in a more symmetrical way 
        for midx in range(1,LL+1):  #Builds LL sets of indices from 1 to LL
            indices.append({ 'vL': offIndices*LL+midx, 'ph': midx, 'vR': offIndices*LL+midx+1 })

        self.LL = LL  
        self.DD = DD  

        self.MPS = inputMPS  
        self.indices = indices
  
        self.chis = mChi
        self.SV = mSV  
        self.SVinv = mSV  
        self.form = 'x'



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
        """ Updates the labels of the physical indices using an input list """
        if len(openIndices) != 2:
            logging.warning("Wrong length of open indices list, doing nothing")
            pass
        else:
            indices = self.indices
        
            indices[0]["vL"] = openIndices[0]
            indices[-1]["vR"] = openIndices[1]
            #self.indices = indices  # Is this even necessary ?
    
    def getEdgeIndices( self ):
        """ Returns a list with the labels of the physical indices """
       
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


        print("Indices:")
        print("")
        print(indicesM, indicesMc)

        toContr = []
        [toContr.append(m) for m in self.MPS]
        [toContr.append(m) for m in MPSconj]

        idxList = indicesM[:]
        idxList.extend(indicesMc)

        # norm = ncon([self.MPS, MPSconj], [indicesM,indicesMc])
        norm = np.real_if_close(ncon(toContr,idxList))

        return norm




        
       
        


# TODO: move as MPS class method 

def bringCan(inpMPS: object, mode: str='LR', epsTrunc: float=1e-10):

    """ Brings input myMPS object to canonical form, returns 'form'.
    According to the mode, we either 
    'L' -> perform a left sweep (and drop final piece so the norm should be 1)
    'LR' -> perform a right sweep after the left one, and truncate the SV below epsTrunc
    'C' -> perform a LR sweep and then bring to Gamma-Lambda-Gamma-Lambda canonical form
    """


    LL = inpMPS.LL
    DD = inpMPS.DD

    chiIn = inpMPS.chis

    MPS = inpMPS.MPS


    chiA = [1]*(LL+1)
    Slist = [np.array([1.])]*(LL+1)
  

    logging.info("Performing a Left Sweep")
    
    Alist = [1.]*LL  # This will hold our A matrices for the LeftCanonical form

    # Shouldn't be necessary if we iniitialize all as 1 
    #chiA[0] = 1
    #Slist[0] = 1.

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

        #print(np.shape(S), np.shape(Vdag), np.shape(MPS[jj]))
        Mtilde = ncon([np.diag(S), Vdag, MPS[jj]], [[-1,1],[1,2],[2,-2,-3]])  

        logging.info(f"Mtilde[{pjj}] = {np.shape(Mtilde)}  - Reshape it as chiA[{pjj}]*d, chiIn[{pjj}")
        Mtr = np.reshape(Mtilde, (chiA[jj]*DD, chiIn[jj+1]))

        U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
        
        logging.info( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
        logging.info(f"chi_{pjj+2} (nonzero SVs) = {np.size(S)}")

        chiA[jj+1] = np.size(S)

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
    #chiA[LL] = 1
    #S[LL] = 1.

    # The last factor should give the normalization

    tail = S @ Vdag 

    normsq = np.real_if_close(tail*np.conjugate(tail))

    if np.abs( normsq - 1.) < 1e-12:
        inpMPS.normalized = 1 
    else:
        inpMPS.normalized = 0 
        if mode == 'L': # Complain only if this is the final step 
            logging.warning(f"MPS not normalized, normsq = {normsq}")


    # So we can just chuck it away I guess..

    print(f"Chis = {chiA}")
    #print(f"SVs = {Slist}")

    inpMPS.MPS = Alist
    inpMPS.chis = chiA
    inpMPS.SV = Slist

    inpMPS.form = 'L'

    

    

    """
    RIGHT SWEEP 
    """

    if(mode == 'LR' or mode == 'C'):
        # Perform a right sweep as well 

        # We will truncate here as well if given a epsTrunc > 0 at input

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

        S = S[ (S >= epsTrunc)]
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
            logging.info(f"Building rightcanon form for M[{pjj}]. First build Mtilde[{pjj}] = M[{pjj}.U.S]")

            Mtilde = ncon([Alist[idx], U, np.diag(S)], [[-1,-2,1],[1,2],[2,-3]])  

            logging.info(f"Mtilde[{pjj}] = {np.shape(Mtilde)}  - Reshape it as chiIn_{pjj} , chiB_{pjj+1}*d")
            Mtr = np.reshape(Mtilde, (chiA[idx],chiB[idx+1]*DD ))

            U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
            
            logging.info( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
            logging.info(f"chi_{pjj} (nonzero SVs) = {np.size(S)}")

            S = S[ (S >= epsTrunc)]
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
        logging.info(f"Last site is just a vector, ie. tensor has shape {np.shape(MPS[0])}")

        Mtilde = ncon([Alist[0], U, np.diag(S)], [[-3,-1,1],[1,2],[2,-2]]) 

        logging.info(f"Mtilde = {np.shape(Mtilde)}")

        # We should still reshape here!
        Mtr = np.reshape(Mtilde, (chiB[0],chiB[1]*DD))

        U, S, Vdag = LA.svd(Mtr,full_matrices=0)  
        
        logging.info( f"SVD: {np.shape(Mtr)} = {np.shape(U)} . {np.shape(S)} . {np.shape(Vdag)}")
        logging.info(f"r[{LL}] (nonzero SVs) = {np.size(S)}")

        # I guess we can reshape the last simply as

        logging.info(f"From {np.shape(Vdag)} ")

        Vdag = np.reshape(Vdag,(chiB[0],DD,chiB[1]))
        logging.info(f"to {np.shape(Vdag)} ")

        Blist[0] = Vdag

        # The last factor should give the normalization

        tail = U @ S

        normsq = np.real_if_close(tail*np.conjugate(tail))

        if np.abs( normsq - 1.) < 1e-12:
            inpMPS.normalized = 1 
        else:
            inpMPS.normalized = 0 
            logging.warning(f"MPS not normalized, normsq = {normsq} (|1-norm| = {np.abs( normsq - 1.)})")


        # So we can just chuck it away I guess..

        print(f"Chis = {chiB}")
        logging.info(f"SVs = {Slist}")

        inpMPS.MPS = Blist
        inpMPS.chis = chiB
        inpMPS.SV = Slist

        inpMPS.form = 'R'


    if(mode == 'C'):  #Build Gamma-Lambda canon form as well

        logging.warning("Building Gamma-Lambda canonical form")
        Glist = [1]*LL
        Glist[0] = Blist[0]

        Sinvlist = Slist[:]

        for ii in range(0,len(Slist)):
            Sinvlist[ii] = [ss**(-1) for ss in Slist[ii]]

        for jj in range(0,LL):
            logging.info(f"Gam[{jj}],{np.shape(Sinvlist[jj-1])},{np.shape(Blist[jj])}")
            Glist[jj] = ncon([np.diag(Sinvlist[jj-1]), Blist[jj]],[[-1,1],[1,-2,-3]])

        #Glist[LL-1] = ncon([np.diag(Sinvlist[LL-2]), Blist[LL-1]],[[-1,1],[1,-2]])
        
        inpMPS.MPS = Glist
        inpMPS.SVinv = Sinvlist

        inpMPS.form = 'C'

    #print(f"Slist = {Slist}")

    return inpMPS.form



def expValOneSite(iMPS: object, oper: np.array, site: int):

    if(iMPS.form != 'LR'):
        bringCan(iMPS,mode='LR',epsTrunc=1e-12)

    conTen = [np.diag(iMPS.SV[1]),np.diag(iMPS.SV[1]),iMPS.MPS[2],np.conj(iMPS.MPS[2]),oper]
    conIdx = [[1,2],[1,3],[3,4,5],[2,6,5],[4,6]]

    return np.real_if_close(ncon(conTen,conIdx))




def checkIdMatrix(ainp: np.array, epstol = 1e-14):
    """Checks if an array is an identity matrix (within machine precision)"""

    a = np.array(ainp)
    if a.shape[0] != a.shape[1]:
        print(f"Not even square: {a.shape}")
        return False
    else:
        size = a.shape[0]
        if np.all(np.abs(a - np.eye(size)) < epstol):
            print(f"identity, size = {size}")
            return True
        else:
            print(f"Square but not id, difference = {np.abs(a - np.eye(size))}")
            return False