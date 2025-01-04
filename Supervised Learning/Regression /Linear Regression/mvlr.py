#Importing our tech stack required
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#using pandas to input my data in
data = pd.read_csv("SupervisedLearning\\Regressions\\Linear Regression\\linear_regression_train.csv")
y = data['Target']
x=data.iloc[:,1:26]
#converting to arrays for solving our transpose functions and simplify mathematics
X =np.array(x)
Y =np.array(y)
m = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_pred = np.dot(X,m)
#calculating our error from concepts learned from Saksham Jain's video
d1 = Y- Y_pred
d2 = Y - Y.mean()
e = 1 - d1.dot(d1)/d2.dot(d2)
print(f"Error:{e}")
r = (e/abs(Y_pred))*100
a = 100-r
print(f"Accuracy:{a}, % Error :{r}")