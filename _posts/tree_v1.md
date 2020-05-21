
Decision Trees (DTs) are a class of supervised learning algorithms for regression and classification tasks. These tasks require the algorithm to learn a mapping from features to a continous or ordinal target. 

DTs utilize a tree based data structure, to recursively partition the dataset on feature values to improve predictive performance in sub-regions. 

A training dataset can be used to grow a tree and obtain decision rules i.e splitting choices. A validation dataset can be used to evaluate the generalizability of the said decision rules.  

For continous targets the sample mean of the dataset is considered the prediction, and for binary targets (1/0) the sample mean is considered the probability of occurance. 

For continous targets, reduction in average RMSE determines splitting feature and value. For binary targets, reduction in average entropy is often used.  

# Tree Structure

* The simplest tree is just one node. 
* A "Node" is a capsule that contains some "Data". 
* Nodes can contain data - string, numbers, array, dicts, etc. 


```
class Node:
    def __init__(self, name, age, occupations):
        self.name = name
        self.age = age
        self.occupations = occupations

root = Node('Donald Trump', 73, ['Real Estate Moghul', 'POTUS']) 
# PS just to clarify, I'm not a trump supporter. 

print(root.name)
print(root.age)
print(root.occupations)
```

    Donald Trump
    73
    ['Real Estate Moghul', 'POTUS']


* Nodes can have children (upto 2 for a binary tree)
* These children also contain data!
* Child nodes can have child nodes too!



```
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def assign(self, left, right):
        self.left = Node(left)
        self.right = Node(right)

root = Node('Donald Trump')
root.assign('Ivanka Trump', 'Trump Jr')
root.left.assign('Ivanka-Girl', 'Ivanka-Boy')

print(root.data)
print('\t', root.left.data)
print('\t\t', root.left.left.data)
print('\t\t', root.left.right.data)
print('\t', root.right.data)
```

    Donald Trump
    	 Ivanka Trump
    		 Ivanka-Girl
    		 Ivanka-Boy
    	 Trump Jr


Tree Traversal: 
* As the tree gets larger, it becomes hard to keep track.
* Special functions must be used to "traverse" or "travel through" the tree.
* Example: the Recursive Print
    * Function starts from the root 
    * Prints the data
    * Checks for the existance of left/right nodes
    * Applies Recursive Print on left/right node if they exist
* Thus Recursive Print traverses the tree and prints data in every node. 
* The Depth parameter can be used, optionally, to count the depth of the current node from the root. 


```
class Node(Node):
    def __init__(self, data):
        super().__init__(data)

    def recursive_print(self, depth=0):

        # Prints Data in Current Node
        print(depth * '\t', '↳', self.data)

        # If Left and Right Node exist, performs the same operation
        if self.left:
            self.left.recursive_print(depth+1)
        if self.right:
            self.right.recursive_print(depth+1)


root = Node('Indira Gandhi')
root.assign('Rajiv Gandhi', 'Sanjay Gandhi')
root.left.assign('Rahul Gandhi', 'Priyanka Gandhi Vadra')
root.left.right.assign('Miraya Vadra', 'Raihan Vadra')
root.recursive_print()
```

     ↳ Indira Gandhi
    	 ↳ Rajiv Gandhi
    		 ↳ Rahul Gandhi
    		 ↳ Priyanka Gandhi Vadra
    			 ↳ Miraya Vadra
    			 ↳ Raihan Vadra
    	 ↳ Sanjay Gandhi


# Regression Tree

We load the Boston Housing Price dataset, which gives us attributes on houses sold. The sale price becomes the target, while the other attributes like size, area, etc. are used as features. There are a total of 506 examples, 1 target and 13 features. [link text](https://)


```
from sklearn.datasets import load_boston
df = load_boston()
feature_names = dict([[i[0]+1, i[1]] for i in enumerate(list(df.feature_names))])
print(df.DESCR[20:1422])
```

    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    



```
import numpy as np
df = np.c_[df.target, df.data] # append Target and Features
print(df.shape) 
```

    (506, 14)


Function to Split Dataset: 
* takes in a dataset, the index of the feature to split on and the feature value to split on. 
* returns all records below and above the feature value as two mutually exclusive datasets.


```
def split(data, idx, value):
    left = data[data[:, idx] <= value]
    right = data[data[:, idx] > value]
    return left, right

left, right = split(df, idx = 12, value = 350)
print(left.shape, right.shape)
```

    (82, 14) (424, 14)


Cost Function:

* takes in dataset, calculates y_hat 
* calculates Root Mean Square Error between y_hat and y


```
def rmse(data):
    y = data[:, 0] 
    y_hat = np.mean(data[:, 0])
    rmse = np.sqrt(np.mean(np.square(y - y_hat)))
    return np.round(rmse, 2)
    
print(rmse(df), rmse(left), rmse(right))
```

    9.19 6.39 9.03


Best Split Finder:

* takes in dataset, iterates over all features and feature values
* at every step, splits dataset into left and right 
* calculates average cost after split
* returns lowest avg cost after split and it's corresponding split index and split value


```
def bestsplit(data):
    best_rmse, best_idx, best_value = rmse(data), None, None
    for idx in range(1, data.shape[1]): # loop over all features 
        x = data[:, idx]
        for value in np.linspace(x.min(), x.max(), 100): # loop over its range of values
            left, right = split(data, idx, value)

            if ((left.shape[0] > 2) & (right.shape[0] > 2)): # check -> left and right datasets must have 1 item or less 

                left_rmse, right_rmse = rmse(left), rmse(right)
                avg_rmse = (left.shape[0] * left_rmse + right.shape[0] * right_rmse)/data.shape[0]

                if avg_rmse <= best_rmse: # select split that yeilds lowest average rmse
                    best_rmse, best_idx, best_value = np.round(avg_rmse, 2), idx, np.round(value, 2)
    return best_rmse, best_idx, best_value


bestsplit(df)
```




    (6.58, 13, 9.78)



Tree Builder: 
* recursive function that finds best split idx and value
* creates left and right nodes
* repeats procedure inside left and right nodes
* until:
    * no best split idx or value is found
    * resulting left and right datasets are too small, less than 2 records
    * resulting tree now exceeds a certain depth

Recursive Print:

* same as before, but slightly tweaked to better print the data contents of a Regression Tree


```
class Node(Node):
    def __init__(self, data):
        super().__init__(data)
        self.idx = None
        self.value = None
        self.y_hat = np.round(np.mean(data[:, 0]), 2)
        self.cost = rmse(data)

    def recursive_build(self, max_depth = 3):
        if max_depth > 0:
            _, self.idx, self.value = bestsplit(self.data)
            if ((self.idx != None) & (self.value != None)): # check 1 -> a best split idx and value must exist 
                left, right = split(self.data, self.idx, self.value)
                self.assign(left, right)
                if ((left.shape[0] > 2) & (right.shape[0] > 2)): # check 2 -> left and right datasets must have 1 item or less
                    self.left.recursive_build(max_depth - 1)
                    self.right.recursive_build(max_depth - 1)

    def recursive_print(self, depth=0):
        feature_names[None] = None
        print(depth * '\t\t', '↳', f'N = {self.data.shape[0]}, y_hat:{self.y_hat}, Cost:{self.cost}')
        print(depth * '\t\t', ' ', f'Split: {feature_names[self.idx], self.value} \n')

        if self.left:
            self.left.recursive_print(depth+1)
        if self.right:
            self.right.recursive_print(depth+1)
```


```
root = Node(df)
root.recursive_build(max_depth = 3)
root.recursive_print()
```

     ↳ N = 506, y_hat:22.53, Cost:9.19
       Split: ('LSTAT', 9.78) 
    
    		 ↳ N = 213, y_hat:29.68, Cost:8.91
    		   Split: ('RM', 6.93) 
    
    				 ↳ N = 142, y_hat:25.35, Cost:5.68
    				   Split: ('DIS', 1.46) 
    
    						 ↳ N = 4, y_hat:50.0, Cost:0.0
    						   Split: (None, None) 
    
    						 ↳ N = 138, y_hat:24.63, Cost:3.89
    						   Split: (None, None) 
    
    				 ↳ N = 71, y_hat:38.34, Cost:7.83
    				   Split: ('RM', 7.42) 
    
    						 ↳ N = 41, y_hat:33.39, Cost:4.65
    						   Split: (None, None) 
    
    						 ↳ N = 30, y_hat:45.1, Cost:6.05
    						   Split: (None, None) 
    
    		 ↳ N = 293, y_hat:17.34, Cost:4.89
    		   Split: ('LSTAT', 14.92) 
    
    				 ↳ N = 130, y_hat:20.71, Cost:3.13
    				   Split: ('RM', 6.73) 
    
    						 ↳ N = 124, y_hat:20.45, Cost:2.69
    						   Split: (None, None) 
    
    						 ↳ N = 6, y_hat:26.0, Cost:5.78
    						   Split: (None, None) 
    
    				 ↳ N = 163, y_hat:14.65, Cost:4.34
    				   Split: ('CRIM', 6.38) 
    
    						 ↳ N = 86, y_hat:17.0, Cost:3.4
    						   Split: (None, None) 
    
    						 ↳ N = 77, y_hat:12.02, Cost:3.73
    						   Split: (None, None) 
    


Interpretation: 
* the first split was done on LSTAT at value 9.78; the left dataset has a prediction of 29.68 while the right has a prediction of 17.34. 
* Thus it means that where a share of low status population is less than 9.78% there the house prices are almost double as compared to elsewhere. 
* Further splits tell you a more granular segregation, and a sufficiently deep regression tree will give you a sharp and accurate predictions for each house. 

Depth: 
* the most important hyperparameter for decision trees
* controls the complexity of the mapping between inputs and outputs
* sufficiently high depth will ensure a perfect mapping between inputs and output, but this will be specific to this dataset and not generalizable to other datasets
* Optimal depth can be selected by observing performance on a validation dataset to ensure our decision rules are generalizable


```

```
