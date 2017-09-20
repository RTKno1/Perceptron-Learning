import numpy as np
import random

class Percept():
    def __init__(self, arr, t):
        self.weights = np.array(arr)
        self.threshold = t
    def set_inputs(self, ins):
        self.i = np.array(ins)
    def sigmoid(self, x):
        return 1 if (1/(1+np.exp(-4*(x - self.threshold)))) >= .7 else 0
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

class HillClimb():
    def evaluate(self, x1, x2, node):
        outarr = np.array([])
        for a in range(2):
            for b in range(2):
                x1.set_value(a)
                x2.set_value(b)
                outarr = np.append(outarr, node.eval())
        return outarr
    def error(self, w):
        x1 = Input()
        x2 = Input()
        node_3 = Percept(w[:2],w[2])
        node_4 = Percept(w[3:5],w[5])
        node_5 = Percept(w[6:8],w[8])
        node_3.set_inputs([x1,x2])
        node_4.set_inputs([x1,x2])
        node_5.set_inputs([node_3, node_4])
        xor = node_5
        #print (np.absolute(np.sum(self.evaluate(x1, x2, xor) - np.array([0, 1, 1, 0]))))
        return np.absolute(np.sum(self.evaluate(x1, x2, xor) - np.array([0, 1, 1, 0])))
    def hillclimb(self, count):
        restart = 0
        l = .8
        w = [ random.uniform(-1, 1) for i in range(9) ]
        while(self.error(w)) > .0001:
            #print(count)
            count += 1
            if restart > 1000: return self.hillclimb(count)
            delta = np.array([ random.uniform(-1, 1) for i in range(9) ]) * l

            if (self.error(w+delta) < self.error(w)): w = w + delta
            else: restart += 1
        return w, count

h = HillClimb()
print(h.hillclimb(0))
