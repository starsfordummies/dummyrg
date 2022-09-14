using TensorOperations
using LinearAlgebra

# Indices ordering: vL, vR, phys

#Base.@kwdef 
mutable struct myMPS
    MPS::Array{Array{ComplexF64,3}}
    LL::Int
    DD::Int
    chis::Vector{Int}

    SV::Vector{Vector{Float64}}
    SVinv::Vector{Vector{Float64}}
    curr_form::Char 

end


function init_MPS(Mlist::Vector{Array{ComplexF64, 3}})
    len = length(Mlist)
    phys_d = size(Mlist[1])[3]
    chis = [size(mj)[1] for mj in Mlist]

    push!(chis, size(last(Mlist))[2])

    SV = [ones(1) for j in 1:len+1]
    SVinv = [ones(1) for j in 1:len+1]

    curr_form = 'x'

    return myMPS(Mlist, len, phys_d, chis, SV, SVinv, curr_form)
end



function truncate_svd(M::Matrix{T}, chiMax::Int=100, epsTrunc::Float64=1e-14) where T <: Union{Float64,ComplexF64}
    F = svd!(M)
    filter!(sv->sv>epsTrunc, F.S)
    cut = min(size(F.S)[1],chiMax)
    return view(F.U,:,1:cut), view(F.S,1:cut), view(F.Vt,1:cut,:), cut
end



function bring_canonical!(inMPS::myMPS, chiMax::Int)

    LL = inMPS.LL
    work = [permutedims(m,(1,3,2)) for m in inMPS.MPS]

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
    inMPS.MPS = [permutedims(m,(1,3,2)) for m in work]
    inMPS.SVinv = [s.^(-1) for s in inMPS.SV]
end


function overlap(bra::myMPS, ket::myMPS, conjugate::Bool = true)::ComplexF64
    # < u | v > 
    if conjugate 
        braU = [conj(u) for u in bra.MPS]
    else
        braU = bra.MPS
    end

    ketV = ket.MPS

    blob = reshape([1.],(1,1))
    for (u, v) in zip(braU, ketV)
        @tensor blob[vRc,vR] := blob[a,b]*v[b,vR,d]*u[a,vRc,d] 
    end

    return tr(blob)

end



function get_norm_zip(inMPS::myMPS)::Float64
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
        


function random_MPS(LL::Int, DD::Int = 2) :: myMPS
    mlist = [rand(ComplexF64,10,10,DD)  for j in 1:LL] 
    mlist[1] = rand(ComplexF64,1,10,DD) 
    mlist[LL] = rand(ComplexF64, 10,1,DD) 

    return init_MPS(mlist) 
end

function product_state(LL::Int)

    chi =1 
    d = 2
    plus = fill(1/sqrt(d),d)
    outMPS = fill(reshape(plus,(chi,chi,d)),LL)

    return outMPS
end

