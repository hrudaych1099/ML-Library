#importing our tech stack required
#learnt concept from neuralnine mathematical explanation video
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("SupervisedLearning\\Regressions\\Linear Regression\\linear_regression_train.csv")
data1 = pd.read_csv("SupervisedLearning\\Regressions\\Linear Regression\\linear_regression_test.csv")
y = data['Target']
x = data['Feature_2']
#I took low learning rate for more accuracy, and 50 iterations of the data
m, c = 0, 0
learning_rate = 0.0001 
epochs = 50
# Finding the Gradient Descent for the Dataset given with Mean Square Error
n = len(x)
for _ in range(epochs):
    y_pred = m * x + c
  # Compute Gradients(differentate mean square errors)
    dm = -(2/n) * np.sum((y - y_pred) * x)
    dc = -(2/n) * np.sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
e = (1/n) * np.mean((y - y_pred)**2)
print(f"Slope: {m}, Intercept: {c}, Error: {e}")
X = np.linspace(np.amin(x),np.max(x),len(y))
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(X, m*X + c, color='red', label='Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
