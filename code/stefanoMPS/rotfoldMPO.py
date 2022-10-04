#from math import sin, cos, sinh, cosh, sqrt
import numpy as np
from scipy import linalg as LA
from myUtils import sncon as ncon

sx = np.asarray([[0., 1], [1, 0.]])
sz = np.asarray([[1, 0.], [0., -1]])


def buildRotFoldMPO(Tmax: float, dt: float, gz: float = 0.2, rules: dict = {"mmode": "svd", "ttype": "real", "fold": False}, LR='R'):
    op = np.asarray([1.,0,0,1])
    return buildRotFoldMPO_op(op, Tmax, dt, gz, rules, LR)


def buildRotFoldMPO_op(op: np.ndarray, Tmax: float, dt: float, gz: float = 0.2, rules: dict = {"mmode": "svd", "ttype": "real", "fold": False}, LR='R'):

    mode = rules["mmode"] 
    time_type = rules["ttype"]
    fold = rules["fold"] 


    if fold:
        LL = round(Tmax/dt)
    else:
        # if unfolded we explicitly go forwards and then backwards, so we double the length
        LL = 2*round(Tmax/dt)
  

    #print(f"MPO length = {LL}, g_z = {gz}")
    if LL == 0 or LL%2 != 0:
        raise ValueError(f"Odd length for the MPO, L={LL}")


    if time_type == "real":
        print(f"Building **REAL** time evol MPO")
    elif time_type == "imag":  # IMAGINARY TIME
        print(f"Building **IMAG** time evol MPO with dtau = i dt")
        dt = -1j*dt  # now the dt to the right is tau, t = -i tau 


    # In principle we can build it in several different ways
    if mode == "sin":
        raise NotImplementedError

    elif mode == "svd":
        print("Using SVD decomposition")

        # We're doing exp(-iHdt), and H = - ZZ - g X 
        Uzz =np.reshape(LA.expm(1j*dt*np.kron(sz,sz)),(2,2,2,2))
        Ux = LA.expm(1j*dt*gz*0.5*sx)
       
        u, s, v = LA.svd( np.reshape(np.transpose(Uzz,(0,2,1,3)),(4,4)) )
       
        # Building u*sqrt(s) and sqrt(s)*v
        vss = LA.sqrtm(np.diag(s[:2])) @ v[:2,:]
        ssu = u[:,:2] @ LA.sqrtm(np.diag(s[:2]))

        # MPO = ncon([np.reshape(ssu,(2,2,2)),np.reshape(vss,(2,2,2))],[[-3,1,-2],[-1,1,-4]]) 
        zzMPO = ncon([np.reshape(ssu,(2,2,2)),np.reshape(vss,(2,2,2))],[[1,-4,-2],[-1,-3,1]]) 

        WW = ncon([Ux, zzMPO, Ux],[[-3,1],[-1,-2,1,2],[2,-4]])

    else:
        raise ValueError("Not clear which mode you want")



    if fold:
        print(f"**Folded MPO with operator {op} ")

        WWfold = ncon([WW,np.conj(WW)],[[-1,-3,-5,-7],[-2,-4,-6,-8]]).reshape(4,4,4,4)

        plus = np.kron([1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),1/np.sqrt(2)]);
        #altplus = np.kron([1,0],[1,0])  (= (1,0,0,0) )
        #one =  np.asarray([1.,0,0,1]);  

        Wbottom = ncon((plus, WWfold), ([1],[-1,-2,-3,1])).reshape(4,4,4,1)
        Wtop = ncon((op, WWfold), ([1],[-1,-2,1,-3])).reshape(4,4,1,4)

        wMPO = [WWfold]*LL
        wMPO[0] = Wbottom
        wMPO[LL-1] = Wtop


    else: 

        # Unfolded 

        # The boundary vector, a prod. state  (for now)
        plus = np.array([1/np.sqrt(2),1/np.sqrt(2)])
        
        """
        We can either rotate first and the contract with the edges to close the MPO
        or do it the other way around - it should make no difference at all.

        Here we first contract, then rotate. 

        """   
        
        # Recall:
        # MPS has legs (vL, vR, pU )
        # MPO has legs (vL, vR, pU, pD)

        # Contract with the boundary vectors to close the MPO
        Wbottom = ncon((plus, WW), ([1],[-1,-2,-3,1])).reshape(2,2,2,1)
        
        wMPO = [WW]*LL
        wMPO[0] = Wbottom
        wMPO[LL-1] = Wbottom


        # For the backwards-time part, we conjugate and rotate 180deg (ie. transpose (1,0,3,2) )

        for jj in range(LL//2, LL):
            #print(f"flip&conj elem {jj}/{LL-1}")
            wMPO[jj] = np.conj(wMPO[jj]).transpose(1,0,3,2)
            #print(np.shape(wMPO[jj]))


    """    2            0
         0   1   />   3   2
           3            1
    """

    # At the end, rotate
    if LR == 'R':  # default, for getting right leading vec
        rMPO = [ w.transpose(3,2,0,1) for w in wMPO ]
    elif LR == 'L': # if we want to get the left dominant vec
        rMPO = [ w.transpose(3,2,1,0) for w in wMPO ]



    return rMPO





def buildMixRotFoldMPO(times: dict, gz: float = 0.2, rules: dict = {"mmode": "svd", "fold": False}):

    # TODO: implement different dt's for real/imag steps? 

    mode = rules["mmode"] 
    fold = rules["fold"] 

    tauMax = times["taumax"]
    dtau = times["dtau"]
    tMax = times["tmax"]
    dt   = times["dt"]
    

    LLim = round(tauMax/dtau)
    LLre = round(tMax/dt)

    LL = LLim+LLre

    if fold:
        print(f"**Folded MPO - imag.time up to tau={tauMax}(dtau={dtau}) and real.time up to t={tMax}(dt={dt})")
        print(f"MPO length = {LL}")
    else:
        print(f"**UNfolded MPO - imag.time up to tau={tauMax}(dtau={dtau}) and real.time up to t={tMax}(dt={dt})")
        print(f"MPO length = {2*LL}")


    if LL == 0 or LL%2 != 0:
        raise ValueError(f"Odd length for the MPO, L={LL}")

    WWre = np.array(0.)
    WWim = np.array(0.)

    # In principle we can build it in several different ways
    if mode == "sin":

        raise NotImplementedError

    elif mode == "svd":
        print("Using SVD decomposition")

        # first build the ***real*** time evol. operator
        # We're doing exp(-iHdt), and H = - ZZ - g X 
        Uzz =np.reshape(LA.expm(1j*dt*np.kron(sz,sz)),(2,2,2,2))
        Ux = LA.expm(1j*dt*gz*0.5*sx)
       
        u, s, v = LA.svd( np.reshape(np.transpose(Uzz,(0,2,1,3)),(4,4)) )
       
        # Building u*sqrt(s) and sqrt(s)*v
        vss = LA.sqrtm(np.diag(s[:2])) @ v[:2,:]
        ssu = u[:,:2] @ LA.sqrtm(np.diag(s[:2]))

        # MPO = ncon([np.reshape(ssu,(2,2,2)),np.reshape(vss,(2,2,2))],[[-3,1,-2],[-1,1,-4]]) 
        zzMPO = ncon([np.reshape(ssu,(2,2,2)),np.reshape(vss,(2,2,2))],[[1,-4,-2],[-1,-3,1]]) 

        WWre = ncon([Ux, zzMPO, Ux],[[-3,1],[-1,-2,1,2],[2,-4]])

        # Now building the ***imaginary*** time evolution op
        dt = -1j*dt  # now the dt to the right is tau, t = -i tau 

        Uzz =np.reshape(LA.expm(1j*dtau*np.kron(sz,sz)),(2,2,2,2))
        Ux = LA.expm(1j*dtau*gz*0.5*sx)
       
        u, s, v = LA.svd( np.reshape(np.transpose(Uzz,(0,2,1,3)),(4,4)) )
       
        # Building u*sqrt(s) and sqrt(s)*v
        vss = LA.sqrtm(np.diag(s[:2])) @ v[:2,:]
        ssu = u[:,:2] @ LA.sqrtm(np.diag(s[:2]))

        # MPO = ncon([np.reshape(ssu,(2,2,2)),np.reshape(vss,(2,2,2))],[[-3,1,-2],[-1,1,-4]]) 
        zzMPO = ncon([np.reshape(ssu,(2,2,2)),np.reshape(vss,(2,2,2))],[[1,-4,-2],[-1,-3,1]]) 

        WWim = ncon([Ux, zzMPO, Ux],[[-3,1],[-1,-2,1,2],[2,-4]])





    if fold:

        WWimfold = ncon([WWim,np.conj(WWim)],[[-1,-3,-5,-7],[-2,-4,-6,-8]]).reshape(4,4,4,4)
        WWrefold = ncon([WWre,np.conj(WWre)],[[-1,-3,-5,-7],[-2,-4,-6,-8]]).reshape(4,4,4,4)

        plus = np.kron([1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),1/np.sqrt(2)]);
        one =  np.asarray([1.,0,0,1]);  

        if LLim > 0: 
            Wbottom = ncon((plus, WWimfold), ([1],[-1,-2,-3,1])).reshape(4,4,4,1)
        else: # real time evol only
            Wbottom = ncon((plus, WWrefold), ([1],[-1,-2,-3,1])).reshape(4,4,4,1)

        if LLre > 0: 
            Wtop = ncon((one, WWrefold), ([1],[-1,-2,1,-3])).reshape(4,4,1,4)
        else: # real time evol only
            Wtop = ncon((one, WWimfold), ([1],[-1,-2,1,-3])).reshape(4,4,1,4)

        wMPO = [WWimfold]*LLim + [WWrefold]*LLre
        wMPO[0] = Wbottom
        wMPO[LL-1] = Wtop


    else: 
        # Unfolded 

        # The boundary vector, a prod. state  (for now)
        plus = np.array([1/np.sqrt(2),1/np.sqrt(2)])
        
        """
        We can either rotate first and the contract with the edges to close the MPO
        or do it the other way around - it should make no difference at all.

        Here we first contract, then rotate. 

        """   
        
        # Recall:
        # MPS has legs (vL, vR, pU )
        # MPO has legs (vL, vR, pU, pD)

        # Contract with the boundary vectors to close the MPO
        if LLim > 0:
            Wbottom = ncon((plus, WWim), ([1],[-1,-2,-3,1])).reshape(2,2,2,1)
        else:
            Wbottom = ncon((plus, WWre), ([1],[-1,-2,-3,1])).reshape(2,2,2,1)


        wMPOim = [WWre]*LLim
        wMPOre = [WWim]*LLre

        wMPO = wMPOim
        wMPO.extend(wMPOre)
        wMPO.extend(wMPOre)
        wMPO.extend(wMPOim)

        wMPO[0] = Wbottom
        wMPO[2*LL-1] = Wbottom


        # For the backwards-time part, we conjugate and rotate 180deg (ie. transpose (1,0,3,2) )

        for jj in range(LL, 2*LL):
            #print(f"flip&conj elem {jj}/{LL-1}")
            wMPO[jj] = np.conj(wMPO[jj]).transpose(1,0,3,2)
            #print(np.shape(wMPO[jj]))



    # At the end, rotate
    rMPO = [ w.transpose(3,2,0,1) for w in wMPO ]



    return rMPO
