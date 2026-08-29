def zero_grad(parameters):
    for p in parameters:
        p.grad=0.0

def SGD(parameters,lr):
    for p in parameters:
        p.data-=lr*p.grad
        