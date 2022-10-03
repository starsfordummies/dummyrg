def aa(x):
    return x

def bb(x):
    return x*x

def callf(ff:callable, x):
    return ff(x)

print(callf(aa,3))
print(callf(bb,3))