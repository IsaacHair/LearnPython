#simple neural network to compute "or"
#has 2 input neurons and a bias neuron that feed to the output neuron
import numpy, random, os, math
lr = 1
bias = 1
weights = [random.random(), random.random(), random.random()]

def Perceptron(input1, input2, output, learn) :
    outputP = input1*weights[0]+input2*weights[1]+bias*weights[2]
    outputP = 1/(1+math.exp(-outputP))
    error = output - outputP
    if learn == 1 :
        weights[0] += error * input1 * lr
        weights[1] += error * input2 * lr
        weights[2] += error * bias * lr
    else :
        return outputP

for i in range(50000) :
    Perceptron(1, 1, 1, 1)
    Perceptron(0, 1, 1, 1)
    Perceptron(1, 0, 1, 1)
    Perceptron(0, 0, 0, 1)

while 1:
    print("Test time (enter a): ")
    a = int(input())
    print("enter b: ")
    b = int(input())
    print("result: ", Perceptron(a, b, 0, 0))
    print("weights: ", weights[0], weights[1], weights[2])
