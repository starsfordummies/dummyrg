# Last modified: 2022/05/12 18:39:51

# Self-dual Ising DMRG code using ITensor library 

if gethostname() == "majorana"
    println("On majorana, using MKL")
    using MKL
    using Plots: plot, plot!, unicodeplots
    unicodeplots()   # We make text-mode plots 
else
    using Plots: plot, plot!
end

using ITensors
using LsqFit
using JLD2


function myBuildHam(sites ; JJ = 1. , lambda = 1. , pp = 0.0)
    println("Building Hamiltonian for J=$(JJ), λ=$(lambda), p=$(pp)")
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
maxdim!(sweeps,10,20,100,200,300,500) # gradually increase states kept
cutoff!(sweeps,1E-12) # desired truncation error


energy, psi = dmrg(H,psi0,sweeps)

println("Bond dimension: $(maxlinkdim(psi))")
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

# Plot S vs log(sin.. )
# If we are critical, according to CFT this should be a 
# straight line, whose slope is the central charge
# TODO: check factors 1/3, 1/6.. 

low = 4
xdatatemp = Array(range(low,N//2-low))
xdata2 = [ log(N/π * sin(π*ell/N)) for ell in xdatatemp]
ydata2 = entropies[low:low+length(xdata2)-1]

plot(xdata2,ydata2)

# Try and fit the central charge
@. mymodel(xx, p) = p[1]*xx + p[2]

#Initial guess for c, offset
p0 = [0.7,0.3]

fit = curve_fit(mymodel, xdata2, ydata2, p0)
println(fit.param)
println(6*fit.param[1])


# We can plot over the full range to ensure that the entropy 
# is actually symmetric around L/2 

xdatatemp3 = Array(range(1,N-1))
xdata3 = [ log(N/π * sin(π*ell/N)) for ell in xdatatemp3]
ydata3 = entropies[1:N-1]

plot(xdata3,ydata3)

energy_psi = inner(psi',H,psi)

# We can save the GS psi on a file 
filename = "psiGS_isingSD_L$(N)_chi$(maxlinkdim(psi)).jld2"
jldsave(filename; psi, N) 

#TODO: we want to save also the model params and all that I guess
