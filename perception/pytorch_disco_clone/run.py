import torch
from torch import FloatTensor
from torch.autograd import Variable
import ipdb
import copy

st = ipdb.set_trace

# Define the leaf nodes
a = Variable(torch.range(5,9), requires_grad=True)

weights = [Variable(torch.range(1,5)*i, requires_grad=True) for i in (2, 5, 9, 7)]

# unpack the weights for nicer assignment
w1, w2, w3, w4 = weights
# st()
b = w1 * a
c = w2 * b

if True:
	c_copy = c.clone()
	c_copy = c_copy.detach()
	c_copy[1:3] = c[1:3]	
	c = c_copy
# c = c.detach()
d =  w3 * c
e =  torch.sum(w4 * d)

L = (10 - e)

L.register_hook(lambda grad: print("l",grad)) 
e.register_hook(lambda grad: print("e",grad)) 
d.register_hook(lambda grad: print("d",grad)) 
c.register_hook(lambda grad: print("c",grad)) 
b.register_hook(lambda grad: print("b",grad)) 
a.register_hook(lambda grad: print("a",grad)) 

# b.register_hook(lambda grad: print(grad)) 

L.backward()
