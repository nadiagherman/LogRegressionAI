from math import exp
from time import sleep

from numpy import argmax
from numpy.linalg import inv
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MyLogisticRegression:
    def __init__(self):
        self.intercept_ = []
        self.coef_ = []

    # use the gradient descent method
    # simple stochastic GD
    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        outputValues = list(set(y))
        # print(outputValues)
        outputValues.sort()
        self.coef_ = [[0.0 for _ in range(1 + len(x[0]))] for _ in outputValues]

        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]

        for epoch in range(noEpochs):

            for i in range(len(x)):  # for each sample from the training data
                for k in outputValues:
                    outputsOneVsAll = [1 if outY == k else 0 for outY in y]
                    # sleep(10000)
                    ycomputed = sigmoid(self.eval(x[i], self.coef_[k]))  # estimate the output
                    crtError = ycomputed - outputsOneVsAll[i]  # compute the error for the current sample
                    for j in range(0, len(x[0])):  # update the coefficients
                        self.coef_[k][j + 1] = self.coef_[k][j + 1] - learningRate * crtError * x[i][j]
                    self.coef_[k][0] = self.coef_[k][0] - learningRate * crtError * 1

        for k in outputValues:
            self.intercept_.append(self.coef_[k][0])
            self.coef_[k] = self.coef_[k][1:]

    def eval(self, xi, coef):
        yi = coef[0]
        for j in range(len(xi)):
            yi += coef[j + 1] * xi[j]
        return yi

    def predictOne(self, element, k):
        yi = self.intercept_[k]
        for xi in range(len(element)):
            yi += self.coef_[k][xi] * element[xi]
        return yi

    def predict(self, inTest):

        outputList = []
        for k in range(len(self.coef_)):
            outputList.append(sigmoid(self.predictOne(inTest, k)))
        return outputList
