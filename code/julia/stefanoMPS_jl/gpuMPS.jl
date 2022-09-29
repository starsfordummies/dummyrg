
using TensorOperations
using LinearAlgebra
using Tullio
using CUDA, CUDAKernels, KernelAbstractions


struct myMPS
    MPS::Vector{CuArray{Float32,3,CUDA.Mem.DeviceBuffer}}
    LL::Int
    DD::Int
    chis::Vector{Int}

    SV::Vector{Vector{Float32}}
    SVinv::Vector{Vector{Float32}}
    #curr_form::Char 

end
#CUDA.Mem.DeviceBuffer
function init_MPS(Mlist::Vector{CuArray{Float32, 3, CUDA.Mem.DeviceBuffer}}) 
    len = length(Mlist)
    phys_d = size(Mlist[1],3)
    chis = [size(mj,1) for mj in Mlist]

    push!(chis, size(last(Mlist),2))

    SV = [ones(1) for j in 1:len+1]
    SVinv = [ones(1) for j in 1:len+1]

    #curr_form = 'x'

    return myMPS(Mlist, len, phys_d, chis, SV, SVinv)
end

function random_mps(LL::Int, DD::Int = 2, T::DataType = Float32) 
    """ Returns a random myMPS object """
    mlist = fill(CUDA.rand(T,10,10,DD),LL)
    mlist[1] = CUDA.rand(1,10,DD) 
    mlist[LL] = CUDA.rand(10,1,DD) 

    return init_MPS(mlist) 
end


function truncate_svd(M, chiMax::Int=100, epsTrunc::Float32=1f-6) 
    F = CUDA.svd(M)
    S = Vector(F.S)
    filter!(sv -> sv>epsTrunc, S)
    cut = min(size(S,1),chiMax)
    return view(F.U,:,1:cut), view(F.S,1:cut), view(F.Vt,1:cut,:), cut
end



function bring_canonical_opt!(inMPS::myMPS, chiMax::Int) 

    LL = inMPS.LL
    #mps = inMPS.MPS
    mps = permutedims.(inMPS.MPS,[(1,3,2)]) 
  

    DD = inMPS.DD
    #mid = ceil(Int, LL/2)
    #qrA = qr!(reshape(mps[mid],(inMPS.chis[mid]*DD,inMPS.chis[mid+1])))

    for (jj, Aj) in enumerate(mps)
        chiL, chiR = size(Aj,1), size(Aj,3)
        qrA = CUDA.qr!(reshape(Aj,(chiL*DD,chiR)))
        chiT = size(qrA.R,1)

        # Update the A[j] and chi[j+1] elements
        inMPS.chis[jj+1] = chiT
        mps[jj] = reshape(Matrix(qrA.Q),(chiL,DD,chiT))

        if jj < LL # don't build next matrix for last element
            An = mps[jj+1]
            @tullio fastmath=false next[vL,ph,vR] := qrA.R[vL,a]*An[a,ph,vR]
            mps[jj+1] = next
        end
    end

    # Now to one R sweep with truncation

    for jj in reverse(eachindex(mps))
        print(jj)
        Aj = mps[jj]
        chiL, chiR = size(Aj,1), size(Aj,3)

        U, S, Vt, chiT = truncate_svd(reshape(Aj,chiL,DD*chiR), chiMax)

        inMPS.chis[jj] = chiT

        mps[jj] = reshape(Vt,(chiT,DD,chiR))

        if jj > 1
            Ap = mps[jj-1]
           @cutensor next[vL,ph,vR] := Ap[vL,ph,a]*U[a,b]*Diagonal(S)[b,vR]
           mps[jj-1] = next
     
        end

    end

    # And one final L sweep 

    for (jj, Aj) in enumerate(mps)
        chiL, chiR = size(Aj,1), size(Aj,3)

        F = CUDA.svd(reshape(Aj,chiL*DD,chiR))
        chiT = length(F.S)
        inMPS.chis[jj+1] = chiT
   
        mps[jj] = reshape(Matrix(F.U),(chiL,DD,chiT))
        #println("Setting A$(jj)")

        if jj < LL
            inMPS.SV[jj+1] = F.S
            #println("setting SV$(jj+1)")
            An = mps[jj+1]
            @tullio next[vL,ph,vR] := Diagonal(F.S)[vL,a]*F.Vt[a,b]*An[b,ph,vR]
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


function overlap(bra::myMPS, ket::myMPS, conjugate::Bool = true) 
    # < u | v > 
    if conjugate 
        braU = conj(bra.MPS) 
    else
        braU = bra.MPS
    end

    ketV = ket.MPS

   
    blob = reshape(ones(Float32),(1,1))
 


    #blob = Array{T,2}
    #blob[1,1] = 1. +0im
    for jj in eachindex(braU, ketV)
        vj = ketV[jj]
        bu = braU[jj]
        @tullio blob2[vRc,vR] := blob[a,b]*vj[b,vR,d]*bu[a,vRc,d] 
        blob = blob2
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
        
