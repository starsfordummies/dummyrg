using ITensors

N = 100
sites = siteinds("S=1/2",N)

ampo = OpSum()
for j=1:N-1
  ampo += 0.5,"S+",j,"S-",j+1
  ampo += 0.5,"S-",j,"S+",j+1
  ampo += "Sz",j,"Sz",j+1
end
H = MPO(ampo,sites)