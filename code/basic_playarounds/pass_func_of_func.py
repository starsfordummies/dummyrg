def f1(x):
    print(2*x)

def f2( ff , x):
    return ff(x)



def f3( func , *args, **kwargs): 
    return func(*args, **kwargs)

x=3.3
f2(f1,x)

f3(f1, x)


def ff1(x: float = 3, y: float = 2.):
    print(x, y, x*y)

def ff3( func , *args, **kwargs): 
    return func(*args, **kwargs)

ff3(ff1, x=2, y=5)
ff3(ff1, 2, 4)

def ff3change_args( func , *args, **kwargs): 
    for yy in range(2,5):
        print("replacing ")
        kwargs['y'] = yy
        return func(*args, **kwargs)

ff3change_args(ff1, x=2, y=5)
ff3(ff1, 2, 4)
