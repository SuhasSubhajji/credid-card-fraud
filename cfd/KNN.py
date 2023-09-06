##------------K-Nearest Neighbors ALGORITHM-------------------##
print('##------------K-Nearest Neighbors ALGORITHM-------------------##')

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('creditcard.csv')

fraud = len(data[data["Class"]==1])
valid = len(data[data["Class"]==0])
total = fraud + valid
#Fraud_percent= (fraud / total)*100
#Valid_percent= (valid / total)*100

print("\nThe number of Fraudulent transactions(Class 1) are:", fraud)
print("\nThe number of Valid transactions(Class 0) are:", valid)
#print("Class 1 percentage = ", Fraud_percent)
#print("Class 0 percentage = ", Valid_percent)
fraud_ratio = (fraud)/ (float(valid) + (fraud))
print('\nThe ratio of Fraudulent transactions is: {}\n'.format(fraud_ratio))

#Performing the classification of the dataset
X=data.drop(['Time','Amount'],axis=1)
y=data['Class']

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Scaling and Normalization
#Scaling: is used attributes are rescaled into the range of 0 and 1. 
#Normalization: is used to rescale each row of data to have a length of 1. 
# Feature Scaling
#fit: when you want to train your model without any pre-processing on the data. Fit means to fit the model to the data being provided.
#transform: when you want to do pre-processing on the data using one of the functions from sklearn.preprocessing. Transform means to transform the data (produce model outputs) according to the fitted model.
#fit_transform(): It's same as calling fit() and then transform() 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Scikit-learn is a free software machine learning library for the Python programming language.
#It features various classification, regression and clustering algorithms including support vector machines
from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Precision Score: {}\n'.format(precision_score(y_test,y_pred)))
print('Recall Score: {}\n'.format(recall_score(y_test,y_pred)))
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,y_pred)))

