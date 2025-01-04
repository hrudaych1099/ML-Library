#importing our tech stack required
import numpy as np
import pandas as pd
#I will Define the KNN Function here
def knn_classify(X, Y, X_test, k=3):
    pred = []
    for testpoint in X_test:
        #Computing Distances using Euclidian Distance(Most used in KNN other than Manhattan and Minkowiski)
        dist = np.sqrt(np.sum((X - testpoint)**2 , axis = 1))
        #Now we neeed to find indices of K-Nearest neighbours in this case 3 (odd no so that we don't get any ambiguities)
        k_indices = np.argsort(dist)[:k]
        k_value = [Y[i] for i in k_indices]
        #Now we have to consider the majority in the nearest neighbors to come to a decision
        pred.append(np.mean(k_value))
    return pred
#Now I iwll import the train and test datas using Pandas
data = pd.read_csv("SupervisedLearning\\Classifications\\KNN\\multi_classification_train.csv")
data1 = pd.read_csv("SupervisedLearning\\Classifications\\KNN\\multi_classification_test.csv")
x = data.iloc[:,1:21]
y = data['Class']
x_test = data1.iloc[:,1:21]
X = np.array(x)
Y = np.array(y)
X_test = np.array(x_test)
pred = knn_classify(X, Y, X_test, k=3)
#print(pred)
#Calculating the F1 score of the model using precision and recall
assert len(Y) == len(pred),
def f1_score(y, y_pred):
    unique_classes = np.unique(y)
    f1_scores = []
    total_samples = len(y)
    weights = []
    for cls in unique_classes:
        tp = np.sum((y_pred == cls) & (y == cls))
        fp = np.sum((y_pred == cls) & (y != cls))
        fn = np.sum((y_pred != cls) & (y == cls))
        #Doing precision and recalll
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        weights.append(np.sum(y == cls) / total_samples)
    weighted_f1 = np.sum(np.array(f1_scores) * np.array(weights))
    return weighted_f1
#calculation of f1 score
f1 = f1_score(Y, np.array(pred))
print(f"F1 Score: {f1}")


