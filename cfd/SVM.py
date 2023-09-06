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
print(cols)
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
from sklearn.metrics import accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,y_pred)))
