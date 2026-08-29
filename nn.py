from .engine import Scalar
import random

class Neuron:
    def __init__(self,nin,activation=Scalar.tanh):
        self.w=[Scalar(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Scalar(random.uniform(-1,1))
        self.activation = activation

    def __call__(self, x):
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        return self.activation(act)

    def parameters(self):
        return self.w+[self.b]



class Layer:
    def __init__(self,nin,nout,activation=Scalar.tanh):
        self.neurons=[Neuron(nin,activation) for _ in range (nout)]

    def __call__(self, x):
        outs=[n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        


class MLP:
    def __init__(self,nin,nouts,activations):
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1],activations[i]) for i in range(len(nouts))]

    def __call__(self,x):
        for Layer in self.layers:
            x=Layer(x)
        return x

    def parameters(self):
        return [p for Layer in self.layers for p in Layer.parameters()]




