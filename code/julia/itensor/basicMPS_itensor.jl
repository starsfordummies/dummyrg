using ITensors 
using Plots


T = randn(4,7,2)

k = Index(4,"index_k")
l = Index(7,"index_l")
m = Index(2,"index_m")

B = ITensor(T,  k,l,m)


U,S,V = svd(B,(k,m),cutoff=1E-2)

Q,R = qr(B,(k,m);positive=true)

#C = combiner(i,k; tags="c")

cutoff = 1E-8
maxdim = 10
T = randomITensor(i,j,k,l,m)
M = MPS(T,(i,j,k,l,m);cutoff=cutoff,maxdim=maxdim)

sites = siteinds(d,N)
cutoff = 1E-8
maxdim = 10
M = MPS(A,sites;cutoff=cutoff,maxdim=maxdim)