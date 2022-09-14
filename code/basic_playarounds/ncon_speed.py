import numpy as np 
from tensornetwork import ncon 


from timeit import default_timer as timer
from datetime import timedelta

""" 
chiL = 30
chiR = 45
Wbond = 3
dd = 2

left = np.random.rand(chiL, chiL, chiL)
wL = np.random.rand(chiL, Wbond, dd, dd)
wR = np.random.rand(Wbond, chiR, dd, dd)
theta = np.random.rand(chiL, dd, dd, chiR)
right = np.random.rand(chiR, chiR, chiR)


# How to do it right 
start = timer()
for i in range(1,10):
    ncon([left, wL, theta, wR, right],
                        [[-1,1,2],[1,7,-2,3],[2,3,6,5],[7,4,-3,6],[-4,4,5]])
end = timer()
print(timedelta(seconds=end-start))

# How to do it wrong
start = timer()
for i in range(1,10):
    ncon([left, wL, theta, wR, right],
                        [[-1,1,9],[1,7,-2,10],[9,10,11,12],[7,4,-3,11],[-4,4,12]])
end = timer()
print(timedelta(seconds=end-start))

# How to do it kinda right but with a lot more code than needed
start = timer()
for i in range(1,10):
    Lwtheta = ncon([left, wL, theta],[[-1,2,3],[2,-2,-3,4],[3,4,-4,-5]])
    wRR = ncon([wR,right], [[-2,2,-4,-5],[-1,2,-3]])
    ncon([Lwtheta,wRR], [[-1,2,-2,3,4],[-4,2,4,-3,3]])

end = timer()
print(timedelta(seconds=end-start))
"""

chiL = 1000 
Dw = 4
rjj = np.random.rand(chiL,Dw,chiL)
bj = np.random.rand(chiL,chiL,2)
wj = np.random.rand(Dw,Dw,2,2)

o = [1,2,3,4,5]
start = timer()
for i in range(1,10):
    ncon([rjj, bj, wj, np.conj(bj)],
           # [[o[0],o[1],o[2]],[-3,o[2],o[3]],[-2,o[1],o[4],o[3]],[-1,o[0],o[4]]])
                           [[4,2,1],[-3,1,3],[-2,2,5,3],[-1,4,5]])
end = timer()
print(timedelta(seconds=end-start))
