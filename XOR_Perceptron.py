#classes defined: Percept and Input(extends Percept)
#parameters: the weights (w_ij) and thresholds (t_j)
import numpy as np
import random

class Percept():
    def __init__(self, arr, t):
        self.weights = np.array(arr)
        self.threshold = t
    def perprint(self):
        return ((i for i in self.i), self.weights, self.threshold)
    def set_inputs(self, ins):
        self.i = np.array(ins)
    def step(self, x):
        return 1 if x > self.threshold else 0
    def eval(self):
        a = []
        for h in self.i:
            a.append(h.eval())
        inputs = np.array(a)
        return self.step(np.dot(inputs, self.weights))
        

class Input(Percept):
    def __init__(self, n = 0):
        self.i = n
    def set_value(self, x):
        self.i = x
    def eval(self):
        return self.i

'''x1 = Input()
x2 = Input()
node_3 = Percept([-1,-2],-3)
node_4 = Percept([1,1],0)
node_5 = Percept([1,2],2)
node_3.set_inputs([x1,x2])
node_4.set_inputs([x1,x2])
node_5.set_inputs([node_3, node_4])
xor = node_5
for a in range(2):
    for b in range(2):
        x1.set_value(a)
        x2.set_value(b)
        print(a,b,xor.eval())'''
