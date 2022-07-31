from math import sqrt

def sqrtifposi(x):
    if x > 0: 
        return sqrt(x)
    else:
        #raise ValueError(f"not good {x} ")
        ValueError(f"not good {x} ")

x = 3 
sqrtifposi(x)

sqrtifposi(-22)

