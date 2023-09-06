#GUI and MySQL
from tkinter import *
import tkinter.messagebox
import mysql.connector
import seaborn as sns

def MAIN():
  R1=Tk()
  R1.geometry('700x500')
  R1.title('WELCOME-1')

  l=Label(R1, text="WELCOME TO CREDIT CARD FRAUD DETECTION PORTAL", font=('algerain',14,'bold'), fg="orange")
  l.place(x=100, y=50)

  b1=Button(R1, text="Register",width=10,height=2,font=('algerain',14), bg="lightblue", fg="red", command=m1)
  b1.place(x=200, y=200)
  
  b2=Button(R1, text="Login",width=10,height=2, font=('algerain',14), bg="lightblue", fg="red", command=m3)
  b2.place(x=200, y=300)
  
  R1.mainloop()


def m1():
  def m2():
    username=e1.get()
    password=e2.get()
    email=e3.get()
    phoneno=e4.get()

    a=mysql.connector.connect(host='localhost', port=3306, user="root", passwd="root", database="creditcard")
    b=a.cursor()
    b.execute("INSERT INTO t1 VALUES(%s,%s,%s,%s)",(username,password,email,phoneno))
    a.commit()

    if e1.get()=="" or e2.get=="":
      tkinter.messagebox.showinfo("SORRY!, PLEASE COMPLETE THE REQUIRED INFORMATION")
    else:
      tkinter.messagebox.showinfo("WELCOME %s" %username, "Lets Login")
      m3()

    
  R2=Tk()
  R2.geometry('600x500')
  R2.title('Register and Login')

  l=Label(R2, text="Login & Register", font=('algerain',14,'bold'), fg="orange")
  l.place(x=200, y=50)

  l1=Label(R2, text="Username", font=('algerain',14), fg="black")
  l1.place(x=100, y=200)
  l2=Label(R2, text="Password", font=('algerain',14), fg="black")
  l2.place(x=100, y=250)
  l3=Label(R2, text="Email", font=('algerain',14), fg="black")
  l3.place(x=100, y=300)
  l4=Label(R2, text="Phoneno", font=('algerain',14), fg="black")
  l4.place(x=100, y=350)
  
  e1=Entry(R2, font=14)
  e1.place(x=200, y=205)
  e2=Entry(R2, font=14, show="**")
  e2.place(x=200, y=255)
  e3=Entry(R2, font=14)
  e3.place(x=200, y=305)
  e4=Entry(R2, font=14)
  e4.place(x=200, y=355)

  b1=Button(R2, text="Signup",width=8,height=1, font=('algerain',14), bg="lightblue", fg="red", command=m2)
  b1.place(x=250, y=400)
      
  R2.mainloop()


def m3():
    def m4():
        a=mysql.connector.connect(host='localhost', port=3306, user="root", passwd="root", database="creditcard")
        b=a.cursor()
        username=e1.get()
        password=e2.get()

        if (e1.get()=="" or e2.get()==""):
            tkinter.messagebox.showinfo("SORRY!, PLEASE COMPLETE THE REQUIRED INFORMATION")
        else:
            b.execute("SELECT * FROM t1 WHERE username=%s AND password=%s",(username,password))

            if b.fetchall():
                tkinter.messagebox.showinfo("WELCOME %s" % username, "Logged in successfully")
                m5()#from function def m5() Function call for Fraud Detection
                
            else:
                tkinter.messagebox .showinfo("Sorry", "Wrong Password")
            
        
    R3=Tk()
    R3.geometry('600x500')
    R3.title('Login')

    l=Label(R3, text="Login", font=('algerain',14,'bold'), fg="orange")
    l.place(x=200, y=50)

    l1=Label(R3, text="Username", font=('algerain',14), fg="black")
    l1.place(x=100, y=200)
    l2=Label(R3, text="Password", font=('algerain',14), fg="black")
    l2.place(x=100, y=250)
      
    e1=Entry(R3, font=14)
    e1.place(x=200, y=205)
    e2=Entry(R3, font=14, show="**")
    e2.place(x=200, y=255)

    b1=Button(R3, text="Login",width=8,height=1, font=('algerain',14), bg="lightblue", fg="red", command=m4)
    b1.place(x=250, y=400)

    R3.mainloop()


def m5():
  R1=Tk()
  R1.geometry('700x600')
  R1.title('WELCOME-2')

  l=Label(R1, text="Algorithm Selection", font=('algerain',18,'bold'), fg="orange")
  l.place(x=150, y=50)

  b1=Button(R1, text="Algorithm Selection",width=18,height=2,font=('algerain',14), bg="lightblue", fg="red", command=algorithm_selection)
  b1.place(x=150, y=250)
  
  b2=Button(R1, text="Bar Graph",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=bar_graph)
  b2.place(x=400, y=250)

  R1.mainloop()
  

def algorithm_selection():
  R1=Tk()
  R1.geometry('1200x700')
  R1.title('WELCOME-3')

  l=Label(R1, text="Algorithms Selection", font=('algerain',20,'bold'), fg="orange")
  l.place(x=450, y=50)

  b1=Button(R1, text="Random Forest",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=Random_Forest)
  b1.place(x=200, y=250)

  b2=Button(R1, text="KNN",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=KNN)
  b2.place(x=400, y=250)

  b3=Button(R1, text="Naive Bayes",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=Naive_Bayes)
  b3.place(x=600, y=250)

  b4=Button(R1, text="SVM",width=14,height=2,font=('algerain',14), bg="lightblue", fg="red", command=SVM)
  b4.place(x=800, y=250)
  
  R1.mainloop()


#SVM algorithm
def Random_Forest():
  print('##------------RANDOM FOREST ALGORITHM-------------------##')

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
  #print(data.head(20))

  #checking the percentage of fraud transactions
  fraud = data[data['Class'] == 1]
  valid = data[data['Class'] == 0]

  print('\nThe number of Fraudulent transactions is: {} \n'.format(len(fraud)))
  print('The number of Valid transactions is: {} \n'.format(len(valid)))

  fraud_ratio = len(fraud)/ float(len(valid) + len(fraud))
  print('The ratio of fraudulent transactions is: {}\n'.format(fraud_ratio))

  #Date Preprocessing
  #lets divide the data into X and y
  columns = data.columns.tolist()
  cols = [c for c in columns if c not in ['Class']]
  target = 'Class'

  X = data[cols]
  y = data[target]


  #Scaling and Normalization
  #scale the data
  from sklearn.preprocessing import StandardScaler  #StandardScaler transforms the data in such a manner that it has mean as 0 and standard deviation as 1
  scaler = StandardScaler()  #Standardize features by removing the mean and scaling to unit variance
  scaled_X = scaler.fit_transform(X)   #fit_transform() is used for the initial fitting of parameters on the training set. Internally, it just calls first fit() and then transform() on the same data.

  #Random Forest model building
  from sklearn.ensemble import RandomForestClassifier   #RandomForestClassifier creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object.  
  from sklearn.model_selection import train_test_split    #train_test_split splits arrays or matrices into random train and test subsets
         
  #doing the train test split (20% test data)
  X_train, X_test, y_train, y_test = train_test_split(scaled_X, y)

  #fit the data to default Random Forest classifier.
  #Model fitting is a measure of how well a machine learning model generalizes(understands and conclude) to similar data to that on which it was trained.
  rfc = RandomForestClassifier() #to fit a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
  rfc.fit(X_train,y_train) 

  #get the predictions
  pred_Random_Forest = rfc.predict(X_test)
  pred_Random_Forest

  #print the precision, recall and  accuracy 
  from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
  print('Precision Score: {}\n'.format(precision_score(y_test,pred_Random_Forest)))
  print('Recall Score: {}\n'.format(recall_score(y_test,pred_Random_Forest)))
  print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_Random_Forest)))



#KNN algorithm    
def SVM():
  print('##------------SVM ALGORITHM-------------------##')

  import numpy as np # linear algebra
  import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  import matplotlib.pyplot as plt
  from sklearn import svm
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  import warnings
  warnings.filterwarnings('ignore')


  data = pd.read_csv("creditcard.csv")


  '''
  Dataset is highly unbalanced and is understandable. 
  Class 0 represents the normal transactions
  Class 1 represents the fraudulent transactions
  '''

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

  #Resampleing the dataset
  data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
  X=data.drop(['Time','Amount'],axis=1)
  y=data['Class']

  # Split the data into training and testing subsets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)

  #Using the rbf kernel to build the initail model.
  classifier= svm.SVC(C= 1, kernel= 'linear', random_state= 0)

  #Fit into Model
  classifier.fit(X_train, y_train)

  #Predict the class using X_test
  y_pred = classifier.predict(X_test)
  y_pred

  from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
  print('Precision Score: {}\n'.format(precision_score(y_test,y_pred)))
  print('Recall Score: {}\n'.format(recall_score(y_test,y_pred)))
  print('Accuracy Score: {}\n'.format(accuracy_score(y_test,y_pred)))



#Naive Bayes algorithm  
def Naive_Bayes():
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


  print("\nThe number of Fraudulent transactions(Class 1) are: ", fraud)
  print("\nThe number of Valid transactions(Class 0) are: ", valid)
  #print("Class 1 percentage = ", Fraud_percent)
  #print("Class 0 percentage = ", Valid_percent)
  fraud_ratio = (fraud)/ (float(valid) + (fraud))
  print('\nThe ratio of fraudulent transactions is: {}\n'.format(fraud_ratio))



  # Importing the dataset
  dataset = pd.read_csv('creditcard.csv')
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

  from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
  print('Precision Score: {}\n'.format(precision_score(y_test,y_pred)))
  print('Recall Score: {}\n'.format(recall_score(y_test,y_pred)))
  print('Accuracy Score: {}\n'.format(accuracy_score(y_test,y_pred)))

  

def KNN():
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


  #Date Preprocessing
  #lets divide the data into X and y
  columns = data.columns.tolist()
  cols = [c for c in columns if c not in ['Class']]
  target = 'Class'

  X = data[cols]
  y = data[target]


  #KNN model building
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import train_test_split

  #doing the train test split 
  X_train, X_test, y_train, y_test = train_test_split(X,y)


  #Scaling and Normalization
  #scale the data
  from sklearn.preprocessing import StandardScaler  #StandardScaler standardize features by removing the mean and scaling to unit variance
  scaler = StandardScaler()
  scaled_X = scaler.fit_transform(X)   #fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.

  #Fit the data to the KNN algorithm
  knn = KNeighborsClassifier(n_neighbors = 5,n_jobs=16)
  knn.fit(X_train,y_train)

  #get the predictions
  pred_KNN = knn.predict(X_test)
  pred_KNN

  from sklearn.metrics import precision_score, recall_score, accuracy_score  #The sklearn.metrics module implements several loss, score, and utility functions to measure classification performance
  print('Precision Score: {}\n'.format(precision_score(y_test,pred_KNN)))
  print('Recall Score: {}\n'.format(recall_score(y_test,pred_KNN)))
  print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_KNN)))



#Bar-graph
def bar_graph():
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  
  data = pd.read_csv('creditcard.csv')
  plt.figure(figsize = (6,5))
  sns.countplot(data['Class'])
  plt.show()

MAIN()

