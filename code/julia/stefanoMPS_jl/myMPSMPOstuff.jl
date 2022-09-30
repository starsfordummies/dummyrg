using TensorOperations
using Tullio

function apply_MPO!(inMPS::myMPS, inMPO::Union{myMPO,myMPOcompact})

    DD = inMPS.DD
    Xas = inMPS.chis
    Xws = inMPO.chis

    for jj in eachindex(inMPS.MPS, inMPO.MPO)
        #@show size(Aj) size(Wj)
        @tensor WAj[vLa,vLw,vRa,vRw,phU] := inMPS.MPS[jj][vLa,vRa,d]*inMPO.MPO[jj][vLw,vRw,phU,d] 
        XL = Xas[jj]*Xws[jj]
        XR = Xas[jj+1]*Xws[jj+1]
        inMPS.MPS[jj] = reshape(WAj,(XL, XR, DD))
        inMPS.chis[jj] = XL

    end

end

""" Applies MPO to psi, using speedup tricks """
function apply_MPO_threaded!(inMPS::myMPS, inMPO::Union{myMPO,myMPOcompact})

    DD = inMPS.DD
    Xas = inMPS.chis
    Xws = inMPO.chis

    Base.Threads.@threads for jj in eachindex(inMPS.MPS, inMPO.MPO)
        #@show size(Aj) size(Wj)
        @tullio WAj[vLa,vLw,vRa,vRw,phU] := inMPS.MPS[jj][vLa,vRa,d]*inMPO.MPO[jj][vLw,vRw,phU,d] 
        XL = Xas[jj]*Xws[jj]
        XR = Xas[jj+1]*Xws[jj+1]
        inMPS.MPS[jj] = reshape(WAj,(XL, XR, DD))
        #inMPS.chis[jj] = XL

    end
    for (jj, Mj) in enumerate(inMPS.MPS)
        inMPS.chis[jj] = size(Mj,1)
    end


end

""" Expectation value <psi|O|psi>  """
function expval_MPO(psi::myMPS, O::Union{myMPO,myMPOcompact})
    psibra = deepcopy(psi)
    psiket = deepcopy(psi)
    println("Check norm: ", get_norm_zip(psi))
    apply_MPO!(psiket,O)
    return overlap(psibra,psiket, true)
    
end

""" Power method - returns psi  """
function power_method(U::myMPO, nIters::Int=10, chiMax::Int=50, faster::Bool=false)
    println("No starting MPS given, starting from a random one with L = $(U.LL) , D=$(U.DD)")
    psi = random_mps(U.LL, U.DD)
    bring_canonical_opt!(psi,20)

    power_method(U, psi, nIters, chiMax, faster)
end

""" Power method - returns psi  """
function power_method(U::myMPO, psi::myMPS, nIters::Int=10, chiMax::Int=50, faster::Bool=false)

    if faster 
        println("faster tricks")
        println("Using $(Base.Threads.nthreads()) threads")
    end

    enPrev = 1e10
    deltaE = 0.
    for jj in 1:nIters
        #println("apply ", jj)
        if faster
            apply_MPO_threaded!(psi, U)
            svd_sweep!(psi,chiMax)
            print(psi.chis)
        else
            apply_MPO!(psi, U)
            bring_canonical!(psi,chiMax)
            print(psi.chis)
        end
        en = expval_MPO(psi, U)
        deltaE = en-enPrev
        #println("deltaE[", jj, "] = ",  en - enPrev)
        enPrev = en 
    end
    @show deltaE 
    bring_canonical!(psi,chiMax)
    return psi
end


""" Test func for benchmarking"""
function test_applyMPO(h::Union{myMPO,myMPOcompact})
    psi = random_mps(10)
    apply_MPO!(psi,h)
end

function test_applyMPOthr(h::Union{myMPO,myMPOcompact})
    psi = random_mps(10)
    apply_MPO_threaded!(psi,h)
end
