module myMPOstuff

using TensorOperations
using LinearAlgebra
#using StaticArrays

using ..myMPSstuff: truncate_svd

# Indices ordering: vL, vR, phU, phD

#Base.@kwdef 

export myMPO, myMPOcompact,
       build_Ising_MPO_compact

struct MPOCompactVector{T <: Number} <: AbstractVector{Array{T,4}}
    Wl :: Array{T,4}
    Ws :: Array{T,4}
    Wr :: Array{T,4}
    len :: Int 
end

Base.size(v::MPOCompactVector) = (v.len,)

Base.IndexStyle(::Type{<:MPOCompactVector}) = IndexLinear()

function Base.getindex(v::MPOCompactVector, i::Int) 
    @assert i >= 0 && i <= v.len
    if i == 1 
        return v.Wl
    elseif i == v.len
        return v.Wr
    else
        return v.Ws
    end
end


struct chisCompactVector <: AbstractVector{Int}
    chi_s :: Int
    len :: Int 
end

Base.size(v::chisCompactVector) = (v.len,)
Base.IndexStyle(::Type{<:chisCompactVector}) = IndexLinear()

function Base.getindex(v::chisCompactVector, i::Int) 
    @assert i >= 0 && i <= v.len

    if i == 1 || i == v.len
        return 1
    else
        return v.chi_s
    end
end


struct myMPO{T <: Number}
    MPO::Vector{Array{T,4}}
    LL::Int
    DD::Int
    chis::Vector{Int}
end

struct myMPOcompact{T <: Number}
    MPO::MPOCompactVector{T}
    LL::Int
    DD::Int
    chis::chisCompactVector
end


function random_mpo(LL::Int, DD::Int = 2) 
    mlist = [rand(ComplexF64,10,10,DD,DD)  for j in 1:LL] 
    mlist[1] = rand(ComplexF64,1,10,DD,DD) 
    mlist[LL] = rand(ComplexF64, 10,1,DD,DD) 

    return mlist 
end


function random_mpo_herm(LL::Int, DD::Int = 2)
    mlist = [rand(ComplexF64,10,10,DD,DD)  for j in 1:LL] 
    mlist[1] = rand(ComplexF64,1,10,DD,DD) 
    mlist[LL] = rand(ComplexF64, 10,1,DD,DD) 

    # TODO: CHECK this 
    mlistC = [m + conj(m).permutedims(1,2,4,3) for m in mlist]
    return mlistC
end


function init_MPO(Mlist::Vector{Array{T, 4}}) where T <: Number
    # vL vR pU pD 
    len = length(Mlist)
    phys_d = size(Mlist[1])[3]
    chis = [size(mj,1) for mj in Mlist]

    push!(chis, size(last(Mlist),2))

    return myMPO(Mlist, len, phys_d, chis)
end



function init_MPOcompact(Mlist::Vector{Array{T, 4}}) where T <: Number
    # vL vR pU pD 
    len = length(Mlist)
    phys_d = size(Mlist[1],3)
    chis_compact = chisCompactVector(size(mj,2),len+1 )
    Mlist_compact = MPOCompactVector(Mlist[1],Mlist[2],Mlist[end],len)

    return myMPOcompact(Mlist_compact, len, phys_d, chis_compact)
end

function init_MPOcompact(len, ww::Tuple{Array{T, 4},Array{T, 4},Array{T, 4}}) where T <: Number
   
    phys_d = size(ww[2],3)
    chis_compact = chisCompactVector(size(ww[1],2),len+1 )
    Mlist_compact = MPOCompactVector(ww[1], ww[2], ww[3], len)

    return myMPOcompact(Mlist_compact, len, phys_d, chis_compact)
end

function IsingMPO(g::Float64, J::Float64 = 1.)

    sx = [0 1 ; 1  0]
    sz = [1 0 ; 0 -1]
    id = I(2)
    zero = zeros(2,2)

    Wl = - reshape(permutedims([g*sx ;;;  J*sz ;;; id],(3,1,2)),(1,3,2,2))
    Ws =  permutedims([id  ;;; zero  ;;; zero ;;;; sz ;;; zero ;;; zero ;;;; g*sx ;;; J*sz ;;; id ],(4,3,1,2))
    Wr = reshape(permutedims([id ;;; sz ;;; g*sx],(3,1,2)),(3,1,2,2))

    return Wl, Ws, Wr 
end


function build_Ising_MPO(LL::Int, g::Float64, J::Float64=1.)
    Wl, Ws, Wr = IsingMPO(g, J)
    Wlist = fill(Ws, LL)
    Wlist[1] = Wl
    Wlist[LL] = Wr
    #@show size(Wl) size(Ws) size(Wr)
    return init_MPO(Wlist)
end


function build_Ising_MPO(LL::Int, g::Float64, J::Float64=1.)
    Wl, Ws, Wr = IsingMPO(g, J)
    Wlist = fill(Ws, LL)
    Wlist[1] = Wl
    Wlist[LL] = Wr
    #@show size(Wl) size(Ws) size(Wr)
    return init_MPO(Wlist)
end

function build_Ising_MPO_compact(LL::Int, g::Float64, J::Float64=1.)

    return init_MPOcompact(LL, IsingMPO(g, J))
end


function expMinusEpsHIsingMPO(LL::Int,  g::Float64 = 0.9, J::Float64 = 1., eps::Float64 = 0.1) 
  
    sx =  [0 1 ; 1  0]
    sz =  [1 0 ; 0 -1]
 

    Ut = reshape(exp(eps .*kron(sz,sz)),(2,2,2,2))
    
    U, S, Vt, chiT = truncate_svd(reshape(permutedims(Ut,(1,3,2,4)),(4,4)) )
   
    #F = svd( reshape(permutedims(Ut,(1,3,2,4)),(4,4)) )
    # only 2 of the 4 SVs are nonzero, so we can truncate 
    #vss = LA.sqrtm(np.diag(s)) @ v;
    vss = sqrt(Diagonal(S)) * Vt
    #ssu = u @ LA.sqrtm(np.diag(s));
    ssu = U * sqrt(Diagonal(S))

    #temp = ncon([reshape(ssu,(2,2,2)),reshape(vss,(2,2,2))],[[-3,1,-2],[-1,1,-4]]) 
    @tensor temp[i,j,k,l] := reshape(ssu,(2,2,2))[k,a,j]*reshape(vss,(2,2,2))[i,a,l]
    #WW= ncon([exp(eps*g*0.5*sx), temp, exp(eps*g*0.5*sx)],[[-3,1],[-1,-2,1,2],[2,-4]])
    @tensor WW[i,j,k,l] := exp(eps*g*0.5*sx)[k,a]*temp[i,j,a,b]*exp(eps*g*0.5*sx)[b,l]

    println(size(WW))
    #WWs = SArray{Tuple{2,2,2,2},Float64,4}(WW)

    # Fill the MPO matrices
    Wmpo = fill(WW,  LL)
    
    Wmpo[1] = reshape(WW[1,1:2,:,:],(1,2,2,2))
    Wmpo[LL] = reshape(WW[1:2,1,:,:],(2,1,2,2))

    return init_MPO(Wmpo)
end

function expMinusEpsHIsingMPO_compact(LL::Int,  g::Float64 = 0.9, J::Float64 = 1., eps::Float64 = 0.1) 
  
    sx =  [0 1 ; 1  0]
    sz =  [1 0 ; 0 -1]
 
    Ut = reshape(exp(eps .*kron(sz,sz)),(2,2,2,2))
    
    U, S, Vt, chiT = truncate_svd(reshape(permutedims(Ut,(1,3,2,4)),(4,4)) )
   
    vss = sqrt(Diagonal(S)) * Vt
    ssu = U * sqrt(Diagonal(S))

    #temp = ncon([reshape(ssu,(2,2,2)),reshape(vss,(2,2,2))],[[-3,1,-2],[-1,1,-4]]) 
    @tensor temp[i,j,k,l] := reshape(ssu,(2,2,2))[k,a,j]*reshape(vss,(2,2,2))[i,a,l]
    #WW= ncon([exp(eps*g*0.5*sx), temp, exp(eps*g*0.5*sx)],[[-3,1],[-1,-2,1,2],[2,-4]])
    @tensor WW[i,j,k,l] := exp(eps*g*0.5*sx)[k,a]*temp[i,j,a,b]*exp(eps*g*0.5*sx)[b,l]

    println(size(WW))
    #WWs = SArray{Tuple{2,2,2,2},Float64,4}(WW)

    # Fill the MPO matrices
    Wmpo = fill(WW,  LL)
    
    Wmpo[1] = reshape(WW[1,1:2,:,:],(1,2,2,2))
    Wmpo[LL] = reshape(WW[1:2,1,:,:],(2,1,2,2))

    return init_MPOcompact(LL, (reshape(WW[1,1:2,:,:],(1,2,2,2)),WW,reshape(WW[1:2,1,:,:],(2,1,2,2))))
end

end #module myMPOstuff
