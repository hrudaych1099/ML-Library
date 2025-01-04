import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("SupervisedLearning\\Regressions\\Polynomial\\polynomial_regression_train.csv")
y = data['Target']
x = data.iloc[:,[1,2,3,4,5]]
X =np.array(x)
Y =np.array(y)
def loss_function(Y, Y_pred): #loss = Mean Square Error basically
    loss = np.mean((Y - Y_pred)**2)
    return loss
def calculate_gradients(X, Y, Y_pred):
    n = len(x)
    dm = (1/n)* np.dot( (X.T),(Y_pred - Y))
    dc = (1/n)*np.sum(Y_pred - Y)
    return dm, dc
def polynomial_feature_set(X, degrees):
    t = X.copy()
    for i in degrees:
        X = np.append(X, t**i, axis=1)
    return X
def train(X, Y, batchsize, degrees, epochs, learning_rate):
    x1 = polynomial_feature_set(X, degrees)
    M, N = x1.shape
    m = np.zeros((N,1))
    c = 0
    Y = Y.reshape(48000,1)
    losses = []
    for epoch in range(epochs):
        for i in range((M-1)//(batchsize+1)):
            starti = i*batchsize
            endi = starti + batchsize
            x_batch = x1[starti : endi]
            y_batch = Y[starti : endi]
            Y_pred = np.dot(x_batch,m) + c
            dm, dc = calculate_gradients(x_batch, y_batch, Y_pred)
            m -= learning_rate*dm
            c -= learning_rate*dc
            l = loss_function(Y, np.dot(x1,m) + c)
            losses.append(1)
            return m, c, losses
def predict(X, m, c, degrees):
    X1 = polynomial_feature_set(X , degrees)
    return np.dot(X1, m) + c
m_train, c_train, losses_train = train(X, Y, batchsize=128, degrees=[3], epochs=200, learning_rate=0.001)
Y_pred = predict(X, m_train, c_train, [3])
d1 = Y - Y_pred
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(f"the r-squared is: {r2}")
"""plt.scatter(X[:,1], Y)
plt.plot(X, Y_pred,color ='red', label='Prediction')
plt.show()"""

    


  