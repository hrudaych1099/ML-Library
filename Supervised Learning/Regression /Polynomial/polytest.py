import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Calcualting the R2 Score
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)
#Using the Polynomial Features 
def polynomial_features(X, degree=2):
    from itertools import combinations_with_replacement

    n_samples, n_features = X.shape
    poly_terms = []

    # Generate combinations of features up to the given degree
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            term = np.prod(X[:, combo], axis=1, keepdims=True)
            poly_terms.append(term)

    return np.hstack(poly_terms)
#using bias
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])
#training
def train_polynomial_regression(X, Y, degree=2):
    #generating poly features
    X_poly = polynomial_features(X, degree)
    X_poly = add_bias(X_poly)
    theta = np.linalg.inv(X_poly.T @ X_poly) @ (X_poly.T @ Y) #Mathematical eqn from Saksham Jain's Mathematical Video of Polynomial Regression
    return theta, X_poly

def predict_polynomial(X, theta, degree=2):
    X_poly = polynomial_features(X, degree)
    X_poly = add_bias(X_poly)
    return X_poly @ theta
if __name__ == "__main__":
    #uploading data given
    data = pd.read_csv("SupervisedLearning\\Regressions\\Polynomial\\polynomial_regression_train.csv")
    x = data.iloc[:,1:6]
    y = data['Target']
    X = np.array(x)
    Y = np.array(y)
    #changing the degree to find best r2 score, degree = 4 gives r2 score of 0.92!
    degree = 4
    theta, X_poly = train_polynomial_regression(X, Y, degree)
    Y_pred = predict_polynomial(X, theta, degree)
    #calculating the R2 score
    r2 = r2_score(Y, Y_pred)
    print(f"R2 Score: {r2}")
    print("Coefficients (theta):", theta)
X_feature = data['Feature_1']
sorted_indices = np.argsort(X_feature)
X_feature_sorted = X_feature[sorted_indices]
Y_sorted = Y[sorted_indices]
Y_pred_sorted = Y_pred[sorted_indices]
plt.scatter(X_feature_sorted, Y_sorted, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X_feature_sorted , Y_pred_sorted, color='red', label='Fitted Curve', linewidth=2)
plt.show()
    



    