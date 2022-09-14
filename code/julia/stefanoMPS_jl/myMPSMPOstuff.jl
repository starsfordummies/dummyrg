

function apply_MPO!(inMPS::myMPS, inMPO::myMPO)

    DD = inMPS.DD
    Xas = inMPS.chis
    Xws = inMPO.chis

    for (jj, (Aj, Wj)) in enumerate(zip(inMPS.MPS, inMPO.MPO))
        #@show size(Aj) size(Wj)
        @tensor WAj[vLa,vLw,vRa,vRw,phU] := Aj[vLa,vRa,d]*Wj[vLw,vRw,phU,d] 
        XL = Xas[jj]*Xws[jj]
        XR = Xas[jj+1]*Xws[jj+1]
        inMPS.MPS[jj] = reshape(WAj,(XL, XR, DD))
        inMPS.chis[jj] = XL
        #if jj == inMPS.LL
        #    inMPS.chis[jj+1] = XR # should unnecessary, it should be 1 ..
        #end
    end
    #@show inMPS.chis

end



function expval_MPO(psi::myMPS, O::myMPO)
    psibra = deepcopy(psi)
    psiket = deepcopy(psi)
    #println("Check norm: ", overlap(psibra,psiket))
    apply_MPO!(psiket,O)
    return overlap(psibra,psiket, true)
    
end



function power_method(U::myMPO, nIters::Int=10, chiMax::Int=50):: myMPS
    psi = random_MPS(U.LL, U.DD)

    enPrev = 1e10
    deltaE = 0.
    for jj in 1:nIters
        #println("apply ", jj)
        apply_MPO!(psi, U)
        bring_canonical!(psi,chiMax)
        en = expval_MPO(psi, U)
        deltaE = en-enPrev
        #println("deltaE[", jj, "] = ",  en - enPrev)
        enPrev = en 
    end
    @show deltaE 
    return psi
end