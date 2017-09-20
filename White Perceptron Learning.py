import numpy as np
import random
from XOR_Perceptron import Percept
from XOR_Perceptron import Input

def step(x):
    return 1 if x > 0 else 0

def classify(x, w):
    return step(np.dot(x, w))

def true_class(x):
    return x[1]

def accuracy_list(ts, w):
    return [abs(true_class(x) - classify(x[0], w)) for x in ts]

def accuracy_rate(ts, w):
    return accuracy_list(ts, w).count(0) / len(ts)

def learn_boolean_function(training_set, num_epochs, lrate = 1):
    weights = np.zeros(len(training_set[0][0]))
    s = 0.0
    while(accuracy_rate(training_set, weights) < .999 and num_epochs > 0):
        for x in training_set:
            f = classify(x[0], weights)
            weights = weights + x[0] * (true_class(x) - f) * lrate
            num_epochs -= 1
    for x in training_set:
        s += abs(true_class(x) - classify(x[0], weights))
    acc = 1 - (s/len(training_set))
    return(weights, acc)

def make_domain(n):
    return [ [ (j>>b) &1 for b in range(n)] for j in range(2**n)]

def make_training_set(domain, b):
    return [ [ np.array(domain[i] + [1]), (b>>i) &1] for i in range(len(domain))]

def main(NUM_VARS):
    miss_count = 0
    count = 0
    dic = {}
    all_counts = []
    D = make_domain(NUM_VARS)
    r = 2 ** 2 ** NUM_VARS
    for x in range(r):
        TS = make_training_set(D,x)
        print(TS)
        fcn_str = "".join([str(row[1]) for row in TS])
        (weights, acc) = learn_boolean_function(TS, num_epochs = 500, lrate = 1)
        if (acc > .999):
            dic[count] = weights
            count += 1
    x1 = Input()
    x2 = Input()
    output = [0,1,1,0]
    total = 0
    for x in dic:
        nodex = Percept(dic[x][:len(dic[x])-1],-dic[x][len(dic[x])-1])
        nodex.set_inputs([x1, x2])
        for y in dic:
            nodey = Percept(dic[y][:len(dic[y])-1],-dic[y][len(dic[y])-1])
            nodey.set_inputs([x1,x2])
            for xy in dic:
                nodexy = Percept(dic[xy][:len(dic[xy])-1],-dic[xy][len(dic[xy])-1])
                nodexy.set_inputs([nodex, nodey])
                outarr = []
                for a in range(2):
                    for b in range(2):
                        x1.set_value(a)
                        x2.set_value(b)
                        outarr.append(nodexy.eval())
                #print (x, nodex.perprint())
                #print (x, y, xy, outarr)
                if outarr == output:
                    total += 1
                    #print (x, y, xy, nodex.perprint(), nodey.perprint(), nodexy.perprint())
    print (total)
main(int(input("Number of vars? ")))
