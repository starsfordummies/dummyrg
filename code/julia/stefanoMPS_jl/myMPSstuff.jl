module myMPSstuff

using TensorOperations
using LinearAlgebra
using Tullio

export myMPS, init_MPS, truncate_svd, random_mps, bring_canonical!, bring_canonical_opt!
export overlap, get_norm_zip, check_norm_SVs, svd_sweep!

# Indices ordering: vL, vR, phys

struct myMPS{T <: Number}
    MPS::Vector{Array{T,3}}
    # MPS::Array{Array{T,3} }
    LL::Int
    DD::Int
    chis::Vector{Int}

    SV::Vector{Vector{Float64}}
    SVinv::Vector{Vector{Float64}}
    #curr_form::Char 

end

function init_MPS(Mlist::Vector{Array{T, 3}}) where T <: Number 
    len = length(Mlist)
    phys_d = size(Mlist[1],3)
    chis = [size(mj,1) for mj in Mlist]

    push!(chis, size(last(Mlist),2))

    SV = [ones(1) for j in 1:len+1]
    SVinv = [ones(1) for j in 1:len+1]

    #curr_form = 'x'

    return myMPS(Mlist, len, phys_d, chis, SV, SVinv)
end

function random_mps(LL::Int, DD::Int = 2, T::DataType = ComplexF64) 
    """ Returns a random myMPS object """
    mlist = fill(rand(T,10,10,DD),LL)
    mlist[1] = rand(T,1,10,DD) 
    mlist[LL] = rand(T, 10,1,DD) 

    return init_MPS(mlist) 
end

function product_state(LL::Int)
    chi =1 
    d = 2
    plus = fill(1/sqrt(d),d)
    outMPS = fill(reshape(plus,(chi,chi,d)),LL)

    return outMPS
end

function truncate_svd(M, chiMax::Int=100, epsTrunc::Float64=1e-14) 
    F = svd!(M)
    filter!(sv -> sv>epsTrunc, F.S)
    cut = min(size(F.S,1),chiMax)
    return view(F.U,:,1:cut), view(F.S,1:cut), view(F.Vt,1:cut,:), cut
end

function truncate_svd_svonly(M, chiMax::Int=100, epsTrunc::Float64=1e-14) 
  
    SVs = svdvals!(M)
    filter!(sv -> sv>epsTrunc, SVs)
    cut = min(size(SVs,1),chiMax)
    return view(SVs,1:cut), cut
end


""" Perform only a L->R sweep truncating at chiMax """
function svd_sweep!(inMPS::myMPS{T}, chiMax::Int) where T <: Number

    LL = inMPS.LL
    DD = inMPS.DD
    
    #println(inMPS.chis)

    inMPS.MPS .= permutedims.(inMPS.MPS, [(1,3,2)] )

    for (jj, Aj) in enumerate(inMPS.MPS)
  
        chiL, chiR = size(Aj,1), size(Aj,3)

        #F = svd!(reshape(Aj,chiL*DD,chiR))
        U, S, Vt, chiT = truncate_svd(reshape(Aj,chiL*DD,chiR), chiMax)
        chiT = length(S)
        inMPS.chis[jj+1] = chiT
   
        inMPS.MPS[jj] = reshape(U,(chiL,DD,chiT))
        #println("Setting A$(jj)")

        if jj < LL
            inMPS.SV[jj+1] = S
            #println("setting SV$(jj+1)")
            @tensor next[vL,ph,vR] := Diagonal(S)[vL,a]*Vt[a,b]*inMPS.MPS[jj+1][b,ph,vR]
            inMPS.MPS[jj+1] = next
        end
    end

    # Put back the canonical form with re-swapped indices (L form)
    
    inMPS.MPS .= permutedims.(inMPS.MPS, [(1,3,2)] )
    
    inMPS.SVinv .= [inv.(s) for s in inMPS.SV]
end


function bring_canonical!(inMPS::myMPS, chiMax::Int)

    LL = inMPS.LL
    work = [permutedims(m,(1,3,2)) for m in inMPS.MPS]
    #println(typeof(work))

    DD = inMPS.DD

    for (jj, Aj) in enumerate(work)
        chiL, chiR = size(Aj)[1], size(Aj)[3]
        qrA = qr!(reshape(Aj,(chiL*DD,chiR)))
        chiT = size(qrA.R)[1]

        # Update the A[j] and chi[j+1] elements
        inMPS.chis[jj+1] = chiT
        work[jj] = reshape(Matrix(qrA.Q),(chiL,DD,chiT))

        if jj < LL # don't build next matrix for last element
            @tensor next[vL,ph,vR] := qrA.R[vL,a]*work[jj+1][a,ph,vR]
            work[jj+1] = next
        end
    end

    # Now to one R sweep with truncation

    for jj in reverse(eachindex(work))
        Aj = work[jj]
        chiL, chiR = size(Aj)[1], size(Aj)[3]

        U, S, Vt, chiT = truncate_svd(reshape(Aj,chiL,DD*chiR), chiMax)

        inMPS.chis[jj] = chiT

        work[jj] = reshape(Vt,(chiT,DD,chiR))

        if jj > 1
           @tensor next[vL,ph,vR] := work[jj-1][vL,ph,a]*U[a,b]*Diagonal(S)[b,vR]
           work[jj-1] = next
     
        end

    end

    # And one final L sweep 

    for (jj, Aj) in enumerate(work)
        chiL, chiR = size(Aj)[1], size(Aj)[3]

        F = svd!(reshape(Aj,chiL*DD,chiR))
        chiT = length(F.S)
        inMPS.chis[jj+1] = chiT
   
        work[jj] = reshape(Matrix(F.U),(chiL,DD,chiT))
        #println("Setting A$(jj)")

        if jj < LL
            inMPS.SV[jj+1] = F.S
            #println("setting SV$(jj+1)")
            @tensor next[vL,ph,vR] := Diagonal(F.S)[vL,a]*F.Vt[a,b]*work[jj+1][b,ph,vR]
            work[jj+1] = next
        else

        end
    end


    # Put back the canonical form with re-swapped indices (L form)
    for jj in eachindex(work) 
        inMPS.MPS[jj] = permutedims(work[jj],(1,3,2))
        inMPS.SVinv[jj] = inMPS.SV[jj].^(-1) 
    end

end


function bring_canonical_opt!(inMPS::myMPS{T}, chiMax::Int) where T <: Number

    LL = inMPS.LL
    #mps = inMPS.MPS
    mps = permutedims.(inMPS.MPS,[(1,3,2)]) 
  

    DD = inMPS.DD
    #mid = ceil(Int, LL/2)
    #qrA = qr!(reshape(mps[mid],(inMPS.chis[mid]*DD,inMPS.chis[mid+1])))

    for (jj, Aj) in enumerate(mps)
        chiL, chiR = size(Aj,1), size(Aj,3)
        qrA = qr!(reshape(Aj,(chiL*DD,chiR)))
        chiT = size(qrA.R,1)

        # Update the A[j] and chi[j+1] elements
        inMPS.chis[jj+1] = chiT
        mps[jj] = reshape(Matrix(qrA.Q),(chiL,DD,chiT))

        if jj < LL # don't build next matrix for last element
            @tensor next[vL,ph,vR] := qrA.R[vL,a]*mps[jj+1][a,ph,vR]
            mps[jj+1] = next
        end
    end

    # Now to one R sweep with truncation

    for jj in reverse(eachindex(mps))
        Aj = mps[jj]
        chiL, chiR = size(Aj,1), size(Aj,3)

        U, S, Vt, chiT = truncate_svd(reshape(Aj,chiL,DD*chiR), chiMax)

        inMPS.chis[jj] = chiT

        mps[jj] = reshape(Vt,(chiT,DD,chiR))

        if jj > 1
           @tensor next[vL,ph,vR] := mps[jj-1][vL,ph,a]*U[a,b]*Diagonal(S)[b,vR]
           mps[jj-1] = next
     
        end

    end

    # And one final L sweep 

    for (jj, Aj) in enumerate(mps)
        chiL, chiR = size(Aj,1), size(Aj,3)

        F = svd!(reshape(Aj,chiL*DD,chiR))
        chiT = length(F.S)
        inMPS.chis[jj+1] = chiT
   
        mps[jj] = reshape(Matrix(F.U),(chiL,DD,chiT))
        #println("Setting A$(jj)")

        if jj < LL
            inMPS.SV[jj+1] = F.S
            #println("setting SV$(jj+1)")
            @tensor next[vL,ph,vR] := Diagonal(F.S)[vL,a]*F.Vt[a,b]*mps[jj+1][b,ph,vR]
            mps[jj+1] = next
        else

        end
    end


    # Put back the canonical form with re-swapped indices (L form)
    for jj in eachindex(inMPS.MPS) 
        inMPS.MPS[jj] = permutedims(mps[jj],[1,3,2])
    end
    inMPS.SVinv .= [inv.(s) for s in inMPS.SV]
end


function overlap(bra::myMPS{T}, ket::myMPS{U}, conjugate::Bool = true) where {T <: Number, U <: Number}
    # < u | v > 
    if U <: Complex && conjugate 
        braU = conj(bra.MPS) 
    else
        braU = bra.MPS
    end

    ketV = ket.MPS

    if T <: Complex || U <: Complex 
        blob = reshape(ones(ComplexF64),(1,1))
    else
        blob = reshape(ones(Float64),(1,1))
    end


    #blob = Array{T,2}
    #blob[1,1] = 1. +0im
    for jj in eachindex(braU, ketV)
        @tensor blob[vRc,vR] := blob[a,b]*ketV[jj][b,vR,d]*braU[jj][a,vRc,d] 
    end

    return tr(blob)

end


function get_norm_zip(inMPS::myMPS)
    normsq = overlap(inMPS, inMPS, true)
    @assert imag(normsq)/real(normsq) < 1e-15 "complex norm?! $normsq"

    return sqrt(real(normsq))
end


function check_norm_SVs(inMPS::myMPS, epsTol::Float64 = 1e-14)::Bool
    sumsqs = fill(false,inMPS.LL+1)
    for (jj, lambdas) in enumerate(inMPS.SV)
        sumsqs[jj] = abs(1 - sum([lam^2 for lam in lambdas])) < epsTol
    end
    return all(sumsqs)
end


function get_entropies(inMPS::myMPS)
    if check_norm_SVs(inMPS, 1e-14) == false
        println("Warning, SVs are not normalized!")
    end

    ents = zeros(inMPS.LL+1)
    for (jj, lambdas) in enumerate(inMPS.SV)
        ents[jj] = sum([-lam^2 *log(lam^2) for lam in lambdas])
    end
    return ents
end
        

end # module myMPSstuff
