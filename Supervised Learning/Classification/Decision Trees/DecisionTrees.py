#importing our tech stacks required 
#learnt concept from Normailised Nerd YT video
from multiprocessing import reduction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing our data which is multi classfication
data = pd.read_csv("SupervisedLearning\\Classifications\\DecisionTrees\\binary_classification_train.csv")
data1 = pd.read_csv("SupervisedLearning\\Classifications\\DecisionTrees\\binary_classification_test.csv")
#defining our nodes
class Node(): # this will act as constructor
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value # this is for leaf node?
class DecisionTree():#this class will be the most important for building the decision tree
    def __init__(self, min_split=2, max_depth=2):
        self.min_split = min_split
        self.max_depth= max_depth
        self.root = None
    def buildtree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        if num_samples>= self.min_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"]>0: #info gain will be given later
                left_subtree = self.buildtree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.buildtree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"],left_subtree, right_subtree, best_split["info_gain"])
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value = leaf_value)
    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf") # we need max infogain to make the nodes correct (hence setting the bar low with -infinity)
        #now we loop all our features to get our split
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    #finding info gain --> we can use gini method or entropy 
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
    def split(self, dataset, feature_index, threshold): #splitting our data (into different nodes)
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent) #weights are nothing but the ratio of data in the daughter node compared to the parent (basically the amount of splitting)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    def print_tree(self, tree=None, indent=" "):#i used this to print my tree 
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    def fit(self, X, Y):#training my tree 
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.buildtree(dataset)
    
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        if tree.value!= None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
#importing data 
X = data.iloc[:, 1:21].values
Y = data['Class'].values
X_test = data1.iloc[:,1:21].values
classifier = DecisionTree(min_split=3, max_depth=2)
classifier.fit(X,Y)
classifier.print_tree()
Y_pred = classifier.predict(X_test) 
def accuracy(Y, Y_pred):
    return np.sum(Y == Y_pred) / len(Y)

acc = accuracy(Y, reduction)
print(acc)





        
    
     







        
