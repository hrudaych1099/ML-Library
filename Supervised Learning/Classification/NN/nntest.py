#importing required data stacks
import numpy as np
import pandas as pd

#defining our sigmoid func
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
#cross entropy function
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#initial parameters for neural network
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)# this is just so that I can use them later
    #w1,b1 are for hidden layer and w2,b2 are for output layers
    W1 = np.random.randn(input_size, hidden_size) * 0.01  
    b1 = np.zeros((1, hidden_size))                     
    W2 = np.random.randn(hidden_size, output_size) * 0.01 
    b2 = np.zeros((1, output_size))                     
    return W1, b1, W2, b2

#forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1) # this is relu propagation 
    Z2 = np.dot(A1, W2) + b2 
    A2 = sigmoid(Z2)        
    return Z1, A1, Z2, A2

Y = Y.reshape(-1, 1)

# Backward Propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]
    
    #grad for output layer 
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    #gradients for hidden layer 
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)#derivative of relu func
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

#calculating f1 score
def f1_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

#neural network training
def train_neural_network(X, Y, hidden_size=64, learning_rate=0.01, epochs=1000):
    input_size = X.shape[1]
    output_size = 1  # Binary classification
    
    # Initialize parameters
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        #forward prop
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = binary_cross_entropy(Y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return W1, b1, W2, b2

#predicting
def predict(X, W1, b1, W2, b2, threshold=0.5):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return (A2 >= threshold).astype(int)

#Now inputting our given dataset
if __name__ == "__main__":
    #using the NN dataset
    data = pd.read_csv("SupervisedLearning\\Classifications\\NN\\nn_train.csv")
    r = 4000
    x = data.iloc[0:r,0:1025]
    y = data['binary_label']
    y = y[0:r]
    X = np.array(x)
    Y = np.array(y) 
    Y = Y.reshape(-1, 1)
    #training the network
    W1, b1, W2, b2 = train_neural_network(X, Y, hidden_size=64, learning_rate=0.001, epochs=1000)
    predictions = predict(X, W1, b1, W2, b2)

    #calculating the f1 score
    f1 = f1_score(Y.flatten(), predictions.flatten())
    print(f"F1 Score: {f1}")
