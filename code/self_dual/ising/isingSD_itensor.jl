using ITensors
using Plots


function myBuildHam(sites ; JJ = 1. , lambda = 1. , pp = 0.0)
    ampo = OpSum()
    for j=1:N-2
        ampo += pp*4, "Sz",j,"Sz",j+2
    end
    for j=1:N-1
    #ampo += 0.5,"S+",j,"S-",j+1
    #ampo += 0.5,"S-",j,"S+",j+1
    ampo += -4*JJ, "Sz",j,"Sz",j+1
    ampo += 4*lambda*pp, "Sx",j,"Sx",j+1
    end
    for j = 1:N
        ampo += -2*lambda, "Sx", j 
    end
    myH = MPO(ampo,sites)
    return myH
end


N = 30
sites = siteinds("S=1/2",N)

# Build the Hamiltonian
H = myBuildHam(sites, JJ = 1. , lambda = 1., pp = 0.)


# Initial random state
psi0 = randomMPS(sites,2)

# DMRG parameters 
sweeps = Sweeps(20) # number of sweeps 
maxdim!(sweeps,10,20,100,200,300,500,100) # gradually increase states kept
cutoff!(sweeps,1E-10) # desired truncation error


energy,psi = dmrg(H,psi0,sweeps)

# Build entropies 

entropies = []
for b = 1:N-1
    orthogonalize!(psi, b)
    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
    SvN = 0.0
    for n=1:ITensors.dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    push!(entropies,SvN)
end
println(entropies)

plot(entropies)