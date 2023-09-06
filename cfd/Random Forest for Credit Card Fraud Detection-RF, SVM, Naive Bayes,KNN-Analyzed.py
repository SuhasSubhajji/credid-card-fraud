##------------RANDOM FOREST ALGORITHM-------------------##
print('##------------RANDOM FOREST ALGORITHM-------------------##')

#Making the imports
import os  #provides functions for interacting with the operating system
import pandas as pd   #pandas used for data manipulation and analysis
import numpy as np   #numpy used for n dimensional arrays and matrices
import matplotlib.pyplot as plt  #matplotlib.pyplot provides a MATLAB-like plotting framework.
import seaborn as sns  #Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import warnings    #to alert the user of some condition in a program, where that condition (normally) doesn’t warrant raising an exception and terminating the program
warnings.filterwarnings('ignore')

#Reading the csv file provided
data = pd.read_csv('creditcard.csv')

#checking the head
#print(data.head(20))

#Checking the percentage of fraud transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

print('\nThe number of Fraudulent transactions is: {} \n'.format(len(fraud)))    #string format() method formats the given string into a nicer output in Python.
print('The number of Valid transactions is: {} \n'.format(len(valid)))

fraud_ratio = len(fraud)/ float(len(valid) + len(fraud))
print('The ratio of fraudulent transactions is: {}\n'.format(fraud_ratio))


#Data Preprocessing(is a technique that is used to convert the raw data into a clean data set)
#Lets divide the data into X and y
#data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
columns = data.columns.tolist()
cols = [c for c in columns if c not in ['Class']]    #Considers all parameters except class parameter
target = 'Class'      #Considers class parameter 
X = data[cols]
y = data[target]

#Scaling and Normalization
#Scaling: is used attributes are rescaled into the range of 0 and 1. 
#Normalization: is used to rescale each row of data to have a length of 1. 
from sklearn.preprocessing import StandardScaler  #StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)   #fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.
#fit: when you want to train your model without any pre-processing on the data. Fit means to fit the model to the data being provided.
#transform: when you want to do pre-processing on the data using one of the functions from sklearn.preprocessing. Transform means to transform the data (produce model outputs) according to the fitted model.
#fit_transform(): It's same as calling fit() and then transform() 

#Random Forest model building
#Doing the train test split (80% train data and 20% test data)
from sklearn.ensemble import RandomForestClassifier    
from sklearn.model_selection import train_test_split    #Import train_test_split function   #train_test_split splits arrays or matrices into random train and test subsets
#Split dataset into training set and test set
#Training set: a subset to train a model.
#Test set: subset to test the trained model.
#X_train, X_test, y_train, y_test = train_test_split(scaled_X, y)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y)

#Fit the data to default Random Forest classifier.
#RandomForestClassifier creates a set of decision trees from randomly selected subset of training set.
#It then aggregates the votes from different decision trees to decide the final class of the test object.
#Fitting is a measure of how well a machine learning model generalizes to similar data to that on which it was trained.
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)  #fit the model on training set 

#Get the predictions of Random Forest model
#Predict the class using X_test
pred_Random_Forest = rfc.predict(X_test)
pred_Random_Forest

#Print the precision, recall and  accuracy 
#Scikit-learn is a free software machine learning library for the Python programming language.
#It features various classification, regression and clustering algorithms including support vector machines
from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Precision Score: {}\n'.format(precision_score(y_test,pred_Random_Forest)))
print('Recall Score: {}\n'.format(recall_score(y_test,pred_Random_Forest)))
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_Random_Forest)))

#Lets plot the graph using seaborn
#Lets try to see it on log scale
plt.figure(figsize = (6,5))
sns.countplot(data['Class'])
#plt.yscale('log')
plt.show()


##------------------NAIVE BAYES ALGORITHM-------------------##

print('##------------------NAIVE BAYES ALGORITHM-------------------##')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#import dataset
data = pd.read_csv("creditcard.csv")

fraud = len(data[data["Class"]==1])
valid = len(data[data["Class"]==0])
total= fraud + valid
#Fraud_percent= (fraud / total)*100
#Valid_percent= (valid / total)*100


print("\nThe number of Fraudulent transactions(Class 1) are: ", fraud)
print("\nThe number of Valid transactions(Class 0) are: ", valid)
#print("Class 1 percentage = ", Fraud_percent)
#print("Class 0 percentage = ", Valid_percent)
fraud_ratio = (fraud)/ (float(valid) + (fraud))
print('\nThe ratio of fraudulent transactions is: {}\n'.format(fraud_ratio))


#DataPreprocessing(is a technique that is used to convert the raw data into a clean data set)
#Lets divide the data into X and y
columns = data.columns.tolist()
cols = [c for c in columns if c not in ['Class']]
target = 'Class'
X = data[cols]
y = data[target]

#Scaling and Normalization
#Scaling: is used attributes are rescaled into the range of 0 and 1. 
#Normalization: is used to rescale each row of data to have a length of 1. 
from sklearn.preprocessing import StandardScaler  #StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)   #fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting Naive Bayes to the Training set
#Fit the data to Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Get the predictions of Naive Bayes model
#Predict the class using X_test
#Predict the response for test dataset
y_pred = classifier.predict(X_test)

#Print the precision, recall and  accuracy 
from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Precision Score: {}\n'.format(precision_score(y_test,y_pred)))
print('Recall Score: {}\n'.format(recall_score(y_test,y_pred)))
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,y_pred)))



##------------KNN ALGORITHM-------------------##

print('##------------KNN ALGORITHM-------------------##')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#reading the dataset
data = pd.read_csv('creditcard.csv')

#checking the percentage of fraud transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

print('\nThe number of Fraudulent cases is: {} \n'.format(len(fraud)))
print('The number of valid transactions is: {} \n'.format(len(valid)))

fraud_ratio = len(fraud)/ float(len(valid) + len(fraud))

print('The ratio of fraudulent transactions is: {}\n'.format(fraud_ratio))


#Data Preprocessing
#lets divide the data into X and y
columns = data.columns.tolist()
cols = [c for c in columns if c not in ['Class']]
target = 'Class'
X = data[cols]
y = data[target]

#Scaling and Normalization
#scale the data
from sklearn.preprocessing import StandardScaler  #StandardScaler standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)   #fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.

#KNN model building
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#doing the train test split 
X_train, X_test, y_train, y_test = train_test_split(X,y)

#Fit the data to the KNN algorithm
#Fitting KNN to the Training set
#Fit the data to Naive Bayes Model
knn = KNeighborsClassifier(n_neighbors = 5,n_jobs=16)
knn.fit(X_train,y_train)

#Get the predictions of KNN model
#Predict the class using X_test
#Predict the response for test dataset
pred_KNN = knn.predict(X_test)

#Print the precision, recall and  accuracy 
from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Precision Score: {}\n'.format(precision_score(y_test,pred_KNN)))
print('Recall Score: {}\n'.format(recall_score(y_test,pred_KNN)))
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_KNN)))




##------------SVM ALGORITHM-------------------##

print('##------------SVM ALGORITHM-------------------##')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  #matplotlib.pyplot provides a MATLAB-like plotting framework.
from sklearn import svm   #SVM library  
from sklearn.svm import SVC  # SVC(Support Vector Classifier) is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data.
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#Reading the csv file provided
data = pd.read_csv("creditcard.csv")

#Checking the percentage of fraud transactions
fraud = len(data[data["Class"]==1])
valid = len(data[data["Class"]==0])
total= fraud + valid
#Fraud_percent= (fraud / total)*100
#Valid_percent= (valid / total)*100

print("\nThe number of Fraudulent transactions(Class 1) are:", fraud)
print("\nThe number of Valid transactions(Class 0) are:", valid)
#print("Class 1 percentage = ", Fraud_percent)
#print("Class 0 percentage = ", Valid_percent)
fraud_ratio = (fraud)/ (float(valid) + (fraud))
print('\nThe ratio of Fraudulent transactions is: {}\n'.format(fraud_ratio))

#Data Preprocessing(is a technique that is used to convert the raw data into a clean data set)

#Lets divide the data into X and y
#data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
columns = data.columns.tolist()
cols = [c for c in columns if c not in ['Class']]    #Considers all parameters except class parameter
target = 'Class'      #Considers class parameter 
X = data[cols]
y = data[target]


#Scaling and Normalization
#scale the data
from sklearn.preprocessing import StandardScaler  #StandardScaler standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)   #fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.

# Split the data into training and testing subsets
from sklearn.model_selection import train_test_split  # Import train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)

#Fit the data to SVM classifier Model
#Create a SVM Classifier
classifier= svm.SVC(C= 1, kernel= 'linear', random_state= 0)
#Train the model using the training sets
classifier.fit(X_train, y_train)

#Get the predictions of SVM model
#Predict the class using X_test
#Predict the response for test dataset
y_pred = classifier.predict(X_test)
y_pred

#Print the precision, recall and  accuracy 
from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Precision Score: {}\n'.format(precision_score(y_test,y_pred)))
print('Recall Score: {}\n'.format(recall_score(y_test,y_pred)))
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,y_pred)))


