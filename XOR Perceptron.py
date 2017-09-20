#classes defined: Percept and Input(extends Percept)
#parameters: the weights (w_ij) and thresholds (t_j)
import numpy as np
import random

class Percept():
    def __init__(self, arr, t):
        self.weights = np.array(arr)
        self.threshold = t
    def set_inputs(self, ins):
        self.i = np.array(ins)
    def sigmoid(self, x):
        return 1 if (1/(1+np.exp(-5*(x - self.threshold)))) >= .7 else 0
    def eval(self):
        a = []
        for h in self.i:
            a.append(h.eval())
        inputs = np.array(a)
        return self.sigmoid(np.dot(inputs, self.weights))
    def circle(self, x, y):
        return 1 if (x**2 + y**2) <= 1 else 0

class Input(Percept):
    def __init__(self, n = 0):
        self.i = n
    def set_value(self, x):
        self.i = x
    def eval(self):
        return self.i

x1 = Input()
x2 = Input()
node1 = Percept([1,0],-1)
node2 = Percept([-1,0],-1)
node3 = Percept([0,1],-1)
node4 = Percept([0,-1],-1)
node5 = Percept([1,1,1,1],3)
node1.set_inputs([x1,x2])
node2.set_inputs([x1,x2])
node3.set_inputs([x1,x2])
node4.set_inputs([x1,x2])
node5.set_inputs([node1,node2,node3,node4])
ucir = node5
total = 0
acc = 0
for a in np.arange(-1.5,1.6,.1):
    for b in np.arange(-1.5,1.6,.1):
        x1.set_value(a)
        x2.set_value(b)
        total += 1
        if (ucir.eval() == ucir.circle(a, b)):
            acc += 1
print (acc/total)
