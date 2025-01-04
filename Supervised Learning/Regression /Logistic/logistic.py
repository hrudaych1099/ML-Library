#Importing Requried tech stacks
#Learnt the concept from AssemblyAI youtube video
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Saving the data
data = pd.read_csv("SupervisedLearning\\Regressions\\Logistic\\binary_classification_train.csv")
x = data.iloc[:, 1:21]
y= data['Class']
X = np.array(x)
Y = np.array(y)
n = len(x)
learning_rate = 0.0001
epochs = 300
#defining the sigmoid function which is what we need in logistic
def sigmoid(z):
    return 1/( 1 + np.exp(-z))
M, N = X.shape
m = np.zeros(N)
c = 0
#finding our weights and bias using mathematics
for _ in range(epochs):
    z = np.dot(X, m) + c
    Y_pred = sigmoid(z)
    dm = (1/n)* np.dot( X.T, (Y_pred-Y))
    dc = (1/n)* np.sum(Y_pred - Y)
    m -= learning_rate*dm
    c -= learning_rate*dc
print(f"Weights : {m}, Bias : {c}")
def predict(X, m, c, threshold=0.5):
    z = np.dot(X, m) + c
    probability = sigmoid(z)
    return (probability >= threshold).astype(int)

predictions = predict(X, m, c)
print(f"Predictions: {predictions}")
#Doing Cross-Entropy loss function - > took this function from a simple video in youtube from Saksham Jain
loss = -np.mean(Y*np.log(Y_pred)+ (1 - Y)*np.log(1-Y_pred))
print(f"Loss : {loss}")
a = np.mean(predictions==Y)
print(f"Accuracy = {a*100}%")
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', label='Data')
X_boundary= np.linspace(np.amin(X),np.max(X),len(Y))
Y_boundary = -(m[0] * X_boundary + c) / m[1]
plt.plot(X_boundary, Y_boundary, color='red', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()










