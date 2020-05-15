from numpy import argmax
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import myLogReg
from normalisation import normalisation

data = load_iris()
inputs = data['data']
outputs = data['target']
outputNames = data['target_names']
featureNames = list(data['feature_names'])

feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
feature2 = [feat[featureNames.index('sepal width (cm)')] for feat in inputs]
feature3 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
feature4 = [feat[featureNames.index('petal width (cm)')] for feat in inputs]
inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('sepal width (cm)')],
           feat[featureNames.index('petal length (cm)')], feat[featureNames.index('petal width (cm)')]] for feat in
          inputs]

import numpy as np

# split data into train and test subsets
np.random.seed(5)
indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
testSample = [i for i in indexes if not i in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]
testInputs = [inputs[i] for i in testSample]
testOutputs = [outputs[i] for i in testSample]
# print(trainInputs)
# print(testInputs)

# normalise the features
trainInputs, testInputs = normalisation(trainInputs, testInputs)
# print(trainInputs)
# print(testInputs)


## LOG REGRESSION WITH TOOL


classifier = linear_model.LogisticRegression(multi_class='ovr')
classifier.fit(trainInputs, trainOutputs)
# print(trainOutputs)
# parameters of the liniar regressor

# print(classifier.coef_)
w0, w1, w2 = classifier.intercept_, classifier.coef_[0], classifier.coef_[1]
for model in zip(w0, w1, w2):
    print('classification model: y(feat1, feat2) = ', model[0], ' + ', model[1], ' * feat1 + ', model[2], ' * feat2')

computedTestOutputs = []
for model in zip(w0, w1, w2):
    computedTestOutputs.append([model[0] + model[1] * el[0] + model[2] * el[1] for el in testInputs])
    # print(computedTestOutputs)
# print(computedTestOutputs)

computedTestOutputs = classifier.predict(testInputs)
# print(computedTestOutputs)

error = 1 - accuracy_score(testOutputs, computedTestOutputs)
print("classification error (tool): ", error)

## LOG REGRESSION WITH MY LOGISTIC REGRESSION

myClassifier = myLogReg.MyLogisticRegression()
myClassifier.fit(trainInputs, trainOutputs)

w0, w1, w2 = myClassifier.intercept_, myClassifier.coef_[0], myClassifier.coef_[1]
#print(myClassifier.coef_)
for model in zip(w0, w1, w2):
    print(' my classification model: y(feat1, feat2) = ', model[0], ' + ', model[1], ' * feat1 + ', model[2], ' * feat2')

myComputedTestOutputs = []
for el in testInputs:
    listOut = myClassifier.predict(el)
    label = argmax(listOut)

    myComputedTestOutputs.append(label)

print("my computed test outputs: " + str(myComputedTestOutputs))
print("actual test outputs: " + str(testOutputs))

error = 0.0
for t1, t2 in zip(myComputedTestOutputs, testOutputs):
    if t1 != t2:
        error += 1
error = error / len(testOutputs)
print("my classification error (manual): ", error)
