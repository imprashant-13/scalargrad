import math

class Scalar:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self.grad=0.0
        self._backward=lambda:None
        self._prev=set(_children)
        self._op=_op
        self.label=label

    def __repr__(self):
        return f"Scalar(data={self.data})"

    def __add__(self, other):
        other=other if isinstance(other,Scalar) else Scalar(other)
        out=Scalar(self.data+other.data,(self,other),'+')

        def _backward():
            self.grad +=1.0*out.grad
            other.grad +=1.0*out.grad
        out._backward=_backward
        return out
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other=other if isinstance(other,Scalar) else Scalar(other)
        out=Scalar(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad +=1.0*other.data*out.grad
            other.grad +=1.0*self.data*out.grad
        out._backward=_backward
        return out

    def __rmul__(self, other):
        return self*other

    def __pow__(self, other):
        assert isinstance(other,(int,float))
        out=Scalar(self.data**other,(self, ),f'**{other}')
        def _backward():
            self.grad+=other*(self.data**(other-1))*out.grad
        out._backward=_backward
        return out

    def __truediv__(self, other):
        return self*other**-1

    def __neg__(self):
        return self*-1

    def __sub__(self, other):
        return self+(-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def exp(self):
        x=self.data
        out=Scalar(math.exp(x),(self, ),'exp')
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    
    def tanh(self):
        t=(math.exp(2*self.data)-1)/(math.exp(2*self.data)+1)
        out=Scalar(t,(self, ),'tanh')

        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        return out

    def relu(self):
        t=max(self.data,0)
        out=Scalar(t,(self, ),'relu')

        def _backward():
            if(t>0):
                self.grad+=1.0*out.grad
            else:
                self.grad+=0
        out._backward=_backward
        return out

    def sigmoid(self):
        t=1/(1+math.exp(-self.data))
        out=Scalar(t,(self, ),'sigmoid')

        def _backward():
            self.grad+=out.data*(1-out.data)*out.grad
        out._backward=_backward
        return out

    def leaky_relu(self,alpha=0.01):
        t=self.data if self.data>0 else alpha*self.data
        out=Scalar(t,(self, ),'leaky relu')
        def _backward():
            if(t>0):
                self.grad+=1.0*out.grad
            else:
                self.grad+=alpha*out.grad
        out._backward=_backward
        return out

    def log(self):
        assert self.data>0
        eps = 1e-12
        x=max(self.data,eps)
        out = Scalar(math.log(x),(self,),"log")
        def _backward():
            self.grad+=(1/(x))*out.grad
        out._backward= _backward
        return out

    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()


