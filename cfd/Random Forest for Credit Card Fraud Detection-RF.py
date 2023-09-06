#making the imports
import os  #provides functions for interacting with the operating system
import pandas as pd   #pandas used for data manipulation and analysis
import numpy as np   #numpy used for n dimensional arrays and matrices
import matplotlib.pyplot as plt  #matplotlib.pyplot provides a MATLAB-like plotting framework.
import seaborn as sns  #Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import warnings  
warnings.filterwarnings('ignore')


#reading the csv file provided
data = pd.read_csv('creditcard.csv')

#checking the head
print(data.head(20))

#Lets try to see it on log scale
plt.figure(figsize = (6,5))
sns.countplot(data['Class'])
#plt.yscale('log')
plt.show()

#checking the percentage of fraud transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

print('\nThe number of Fraudulent cases is: {} \n'.format(len(fraud)))
print('The number of valid transactions is: {} \n'.format(len(valid)))

fraud_ratio = len(fraud)/ float(len(valid) + len(fraud))
print('The ratio of fraudulent transactions is: {}\n'.format(fraud_ratio))

#Date Preprocessing
#lets divide the data into X and y
columns = data.columns.tolist()
cols = [c for c in columns if c not in ['Class']]
print(cols)
target = 'Class'

X = data[cols]
y = data[target]


#Scaling and Normalization
#scale the data
#The process of scaling of the dataset of the attributes of varying scale is called as scaling.
#Normalization is used to rescale each row of data to have a length of 1. It is rescaled into the range of 0 and 1. 

from sklearn.preprocessing import StandardScaler  #StandardScaler standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)   #fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.

#Random Forest model building
from sklearn.ensemble import RandomForestClassifier   #RandomForestClassifier creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object.  
from sklearn.model_selection import train_test_split    #train_test_split splits arrays or matrices into random train and test subsets
       
#doing the train test split (20% test data)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y)

#fit the data to default Ramdom Forest classifier.
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

#get the predictions
pred_Random_Forest = rfc.predict(X_test)
pred_Random_Forest

#print the precision, recall and  accuracy 
from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Precision Score: {}\n'.format(precision_score(y_test,pred_Random_Forest)))
print('Recall Score: {}\n'.format(recall_score(y_test,pred_Random_Forest)))
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_Random_Forest)))



