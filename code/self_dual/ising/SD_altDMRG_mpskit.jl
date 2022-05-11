# Last modified: 2022/05/11 15:59:18

# Code for the time evolution of a **finite** Ising chain 

using MPSKit,MPSKitModels,TensorKit,LinearAlgebra
using Plots: plot, plot!
using ZChop
#using ProgressMeter
#using JLD2

using LsqFit


# Conventions for spin matrices (they seem to carry a 1/2 here )
#println( nonsym_spintensors(1//2 ))

# function nonsym_ising_ham(;J = -1,spin = 1//2,lambda = 0.5,longit=0.0)
#     (sx,sy,sz)=nonsym_spintensors(spin);
#     id = one(sx);

#     hamdat = Array{Union{Missing,typeof(sx)},3}(missing,1,3,3)
#     hamdat[1,1,1] = id;
#     hamdat[1,end,end] = id;
#     hamdat[1,1,2] = J*sz;
#     hamdat[1,2,end] = sz;
#     hamdat[1,1,end] = lambda*sx+longit*sz;

# The Ising MPO Hamiltonian I would expect is 
#
#
#   H = [ 1          ]
#       [ Z          ]
#       [ λX  JZ   1 ]

# Instead here we have

#[ 1    JZ    λX ]                [1           ]
#[            Z  ]  = transpose   [JZ          ]
#[            1  ]                [λX    Z    1]

# It's like rotating counter-clockwise twice? 

function isingSD_ham(; J = 1 ,spin = 1//2,lambda = 1, p=0)
    (sx,sy,sz)= 2 .*nonsym_spintensors(spin);
    id = one(sx);

    hamdat = Array{Union{Missing,typeof(sx)},3}(missing,1,5,5)
    hamdat[1,1,1] = id;
    hamdat[1,1,2] = p*lambda*sx;
    hamdat[1,1,3] = p*sz;
    hamdat[1,1,4] = -J*sz;
    hamdat[1,1,5] = -lambda*sx;
    hamdat[1,2,5] = sx;
    hamdat[1,3,4] = id;
    hamdat[1,4,5] = sz;
    hamdat[1,5,5] = id;
   

    ham = MPOHamiltonian(hamdat);

    return ham
end


#writeTemp = false

myChiMax = 100

#lambda0 = cot( π / 8 )

# To get the critical point we likely need to set J=4, lambda=2 
ham = isingSD_ham(J = 1., lambda = 1. , p = 0.3 );
#quenching_ham = nonsym_ising_ham(lambda = 1.);

len = 30;
startBond = 50;

init = FiniteMPS(rand,ComplexF64,len, ℂ^2,ℂ^startBond);

bonds = [ size(mm.data)[1] for mm in init.CR[1:end] ] 

println("Starting bond dims $(bonds)")
println("First, find ground state")

(ψ0n,_) = find_groundstate(init,ham,DMRG2((trscheme=truncbelow(1e-10))));
#(ψ0n,_) = find_groundstate(ψ0,ham,DMRG());
#(ψ0n,_) = find_groundstate(init,ham,DMRG());

bonds = [ size(mm.data)[1] for mm in ψ0n.CR[1:end-1] ] 
println("Found GS with bond dimension $(bonds)")

# Energy

ene= sum(expectation_value( ψ0n ,ham))

println("Energy = $(zchop(ene))")
# Entropies 
ents = []
Sdiags = []

# Calculate entropies 
for c in ψ0n.CR[1:end-1]
    (U,S,V) = tsvd(c);
    S_array = diag(convert(Array,S));
    #print(axes(S_array))
    #print(S_array)
    #readline()
    #S_diag = diag(S_array);
    #push!(Sdiags, S_diag)
    ent = 0.
    for si in S_array
        ent += - si^2 * log(si^2)
    end
    push!(ents, ent)
    #push!(ents, -tr(S_diag*S_diag'*log(S_diag*S_diag')))
end
println("")
println(ents)
#println(Sdiags)

if isinteractive()
    plot(ents, label = "S")
else
    display( plot(ents, label = "S") )
    println("Not interactive: press enter to close plot")
    readline()
end
#print("Interactive: $(isinteractive())")
#readline()

# Try and fit the central charge
@. mymodel(x, p) = p[1]/6*log(2*len/π * sin(π*x/len)) + p[2]

#Initial guess for c, offset
p0 = [0.7,0.3]

xdata = Array(range(1.5,len))
ydata = ents

fit = curve_fit(mymodel, xdata, ydata, p0)
println(fit.param)

fittedmodel(x)  = fit.param[1]/6*log(2*len/π * sin(π*x/len)) + fit.param[2]
plot!(fittedmodel, 1,29)