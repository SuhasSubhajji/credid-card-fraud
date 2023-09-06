##------------------NAIVE BAYES ALGORITHM-------------------##
print('##------------------NAIVE BAYES ALGORITHM-------------------##')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score, precision_recall_curve
#from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from mlxtend.plotting import plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("creditcard.csv")


fraud = len(data[data["Class"]==1])
valid = len(data[data["Class"]==0])
total= fraud + valid
#Fraud_percent= (fraud / total)*100
#Valid_percent= (valid / total)*100


print("\n\nThe number of fraudulent transactions(Class 1) are: ", fraud)
print("\n\nThe number of normal transactions(Class 0) are: ", valid)
#print("Class 1 percentage = ", Fraud_percent)
#print("Class 0 percentage = ", Valid_percent)
fraud_ratio = (fraud)/ (float(valid) + (fraud))
print('\n\nThe ratio of fraudulent transactions is: {}\n'.format(fraud_ratio))



# Importing the dataset
#dataset = pd.read_csv('creditcard.csv')
X=data.drop(['Time','Amount'],axis=1)
y=data['Class']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
print('Precision Score: {}\n'.format(precision_score(y_test,y_pred)))
print('Recall Score: {}\n'.format(recall_score(y_test,y_pred)))
print('Accuracy Score: {}\n'.format(accuracy_score(y_test,y_pred)))




