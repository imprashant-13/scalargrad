# ScalarGrad
A tiny autograd engine and neural net library built from scratch on scalar values.No NumPy,no PyTorch,just plain Python and math.The whole point is being able to read every line and actually see how backprop works instead of trusting a black box.

The core Scalar engine follows Andrej Karpathy's original micrograd design,reverse mode autodiff over a dynamically built computation graph.On top of that base this repo adds a few extra pieces.

sigmoid activation with a numerically stable gradient computed as out.data times one minus out.data.

leaky_relu with a configurable negative slope,alpha defaults to 0.01.

log with epsilon clamping at 1e-12 so you don't blow up on log of zero.

Per layer activations in MLP,so each layer can use a different activation function instead of one activation for the whole network.

A loss.py module with MSELoss,BCELoss for binary cross entropy,and CELoss for categorical cross entropy with softmax built in.

An optimizer.py module with a plain SGD step function and a zero_grad helper so you're not rewriting that boilerplate every time.

## Installation
This isn't published anywhere,it's just a local package.Drop the folder into your project and import from it.
Project layout:

```
your_project/
    scalargrad/
        __init__.py
        engine.py
        nn.py
        loss.py
        optimizer.py
    train.py
```
Import what you need:

```python
from scalargrad import (
    Scalar,
    Neuron, Layer, MLP,
    MSELoss, BCELoss, CELoss,
    SGD, zero_grad,
)
```

## Quick example
Training a small MLP on a toy regression problem with MSELoss and SGD.
```python
from scalargrad import Scalar, MLP, MSELoss, SGD, zero_grad

xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],]
ys = [1.0,-1.0,-1.0,1.0]

mlp = MLP(
    nin=3,
    nouts=[4,4,1],
    activations=[Scalar.tanh, Scalar.tanh, Scalar.tanh],)

for step in range(50):
    y_pred=[mlp(x) for x in xs]
    loss=MSELoss(y_pred, ys)
    zero_grad(mlp.parameters())
    loss.backward()
    SGD(mlp.parameters(), lr=0.05)

    if (step%10)==0:
        print(f"step {step:2d} | loss = {loss.data:.4f}")
print("final predictions:", [round(p.data, 3) for p in y_pred])
print("targets:          ", ys)
```

Output looks something like this.

```
step  0 | loss = 1.8421
step 10 | loss = 0.6203
step 20 | loss = 0.2114
step 30 | loss = 0.0891
step 40 | loss = 0.0402

final predictions: [0.912, -0.887, -0.901, 0.895]
targets:           [1.0, -1.0, -1.0, 1.0]
```

## How it works
Every Scalar keeps track of its data,its grad which gets filled in during backward,and the operation plus the inputs that created it.Calling backward on the final loss builds a topological ordering of the graph and walks it in reverse,applying the chain rule one node at a time.That's really all backprop is.
```python
a=Scalar(2.0)
b=Scalar(-3.0)
c=a*b+a**2
c.backward()
print(a.grad)
print(b.grad)
```

## Credits
Built on top of Andrej Karpathy's micrograd.Go read the original repo,it's a great teaching tool: https://github.com/karpathy/micrograd

## License
MIT License

Copyright (c) 2026 Prashant Kumar
Permission is hereby granted,free of charge,to any person obtaining a copy
of this software and associated documentation files (the "Software"),to deal
in the Software without restriction,including without limitation the rights
to use,copy,modify,merge,publish,distribute,sublicense and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND EXPRESS OR
IMPLIED,INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,DAMAGES OR OTHER
LIABILITY,WHETHER IN AN ACTION OF CONTRACT,TORT OR OTHERWISE,ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
