# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:03:43 2020

@author: Hamza
"""



import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#importing the data#
train1 = pd.read_csv("train.csv")

#removing unecessary columns
train1 = train1.drop(columns=["Loan_ID"])
X_train, X_test, y_train, y_test= train_test_split(train1.iloc[:, 0:11], train1.iloc[:,11], random_state = 42)

#deleting the null value of columns for simplicity, this removes 20% training values#
#train = train1.dropna()

#finding values to fill the null cells in the data
commonGender = X_train["Gender"].mode().values[0]
commonMStatus = X_train["Married"].mode().values[0]
commonDepend = X_train["Dependents"].mode().values[0]
commonEmployed = X_train["Self_Employed"].mode().values[0]
commonCredit = X_train["Credit_History"].mode().values[0]

meanLoan = X_train["LoanAmount"].mean()
meanLoan = round(meanLoan)
meanTerm = X_train["Loan_Amount_Term"].mean()
meanTerm = round(meanTerm)


#filling in the missig values
X_train["Gender"].fillna(commonGender, inplace=True)
X_train["Married"].fillna(commonMStatus, inplace=True)
X_train["Dependents"].fillna(commonDepend, inplace=True)
X_train["Self_Employed"].fillna(commonEmployed, inplace=True)
X_train["Loan_Amount_Term"].fillna(meanTerm, inplace=True)
X_train["Credit_History"].fillna(commonCredit, inplace=True)
X_train["LoanAmount"].fillna(meanLoan, inplace=True)


#label encoding by using get dummies, a form on one hot encoding
X_train = pd.get_dummies(data=X_train, columns=["Gender","Married","Dependents","Education","Self_Employed"
                                              ,"Credit_History", "Property_Area"])
y_train = pd.get_dummies(data=y_train, drop_first=True )
y_train.rename(columns={"Y":"Loan_Status"},inplace=True)

#prepocessing the test data
testGender = X_test["Gender"].mode().values[0]
testDepend = X_test["Dependents"].mode().values[0]
testEmployed = X_test["Self_Employed"].mode().values[0]
testCredit = X_test["Credit_History"].mode().values[0]
meanLoantest = X_test["LoanAmount"].mean()
testTerm = X_test["Loan_Amount_Term"].mean()
meanLoantest = round(meanLoantest)
testTerm = round(testTerm)

X_test["Gender"].fillna(testGender, inplace=True)
X_test["Dependents"].fillna(testDepend, inplace=True)
X_test["Self_Employed"].fillna(testEmployed, inplace=True)
X_test["Loan_Amount_Term"].fillna(testTerm, inplace=True)
X_test["Credit_History"].fillna(testCredit, inplace=True)
X_test["LoanAmount"].fillna(meanLoantest, inplace=True)

X_test = pd.get_dummies(data=X_test, columns=["Gender","Married","Dependents","Education","Self_Employed"
                                              ,"Credit_History", "Property_Area"])
y_test = pd.get_dummies(data=y_test, drop_first=True )
y_test.rename(columns={"Y":"Loan_Status"},inplace=True)


#training the Random Forest model
clf = RandomForestClassifier(max_depth =11 ,n_estimators = 1000,min_samples_split = 2, min_samples_leaf = 1, 
                             random_state = 42, n_jobs= -1)
clf.fit(X_train, y_train.values.ravel())
clfYPred = clf.predict(X_test) 

print("random forest accuracy on training data {:.2f} ".format(clf.score(X_train, y_train)))
print("random forest accuracy on test data {:.2f}".format(clf.score(X_test, y_test)))



#SVC model
#The code for scaling the data was from :
#Introduction to Machine Learning with Python A Guide for Data Scientists by Andreas C. Müller, Sarah Guido 

min_on_training = X_train.min(axis=0)
# compute the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)
# subtract the min, and divide by range
# afterward, min=0 and max=1 for each feature
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC(C=0.1, gamma=0.1) #these parameters give the best results but is similar to a linear model
svc.fit(X_train_scaled, y_train.values.ravel())
svcYPred = svc.predict(X_test)

print("SVC accuracy on training data {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("SVC accuracy on test data {:.2f}".format(svc.score(X_test_scaled, y_test)))


#Linear model
#Instantiating the Linear model and training the model
lin = LogisticRegression()
lin.fit(X_train, y_train.values.ravel())
linYpred = lin.predict(X_test)

print("Logistic regression accuracy on the training data {:.2f}".format(lin.score(X_train, y_train)))
print("Logistic regression accuracy on the test data {:.2f}".format(lin.score(X_test, y_test)))
