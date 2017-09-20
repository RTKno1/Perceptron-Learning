import numpy as np
import math
import random


##1: function PERCEPTRON LEARN(training set)
##2:    ~w = random vector
##3:    for each epoch do
##4:        for each (~xi, f(~xi) in training set do
##5:            f∗ = A(~xi . ~w)
##6:            ~w = ~w + [λ(f(~xi) − f∗)]~xi
##7:        end for
##8:    end for
##9:    accuracy = 1 −Pi|f(~xi) − A(~xi · ~w)|/len(training set)
##10:   Return ( ~w, accuracy)
##11:   end function

def perceptron_learning(training_set):
    w = np.random.rand(len(training_set[0]))
    count = 0
    acc = 0
    while count < 1000000:
        for l in training_set:
            f = np.dot(l[0], w[training_set.index(l)])
            w = w + l[0]*(l[1] - f)
        count += 1
    for l in training_set:
        s += l[1] - dot(l[0], w)
    acc = 1 - (s/len(training_set))
    return (w, acc)

training_set = []
n = int(2)
for r in range(2**(2**n)):
    i = np.array([([0]*(n+1)),0])
    i[0][n-1] = 1
    training_set.append(i)
for l in training_set[0]:
    for a in range(2):
        for b in range(2):
            for ans in range(2):
                newL = [[a, b, 1], ans]
                training_set[0][training_set.index(l)] = newL
                print(perceptron_learning(training_set))

