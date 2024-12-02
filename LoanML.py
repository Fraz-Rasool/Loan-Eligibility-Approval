import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
df = pd.read_csv("LoanApprovalPrediction.csv")
df.head()
df['loan_amount_log'] = np.log(df['LoanAmount'])
df['loan_amount_log'].hist(bins=20)
df.info()
df.isnull().sum()
df['Total_income'] = np.log(df['ApplicantIncome'] + df['CoapplicantIncome'])
df['Total_income'].hist(bins=20)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loan_amount_log = df.loan_amount_log.fillna(df.loan_amount_log.mean())
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df.isnull().sum()
x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y = df.iloc[:,12].values
x
y
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state= 0)
from sklearn.preprocessing import LabelEncoder
Labelencoder_x = LabelEncoder()
from sklearn.preprocessing import LabelEncoder
Labelencoder_x = LabelEncoder()
for i in range(0, 5):
    X_train[:, i] = Labelencoder_x.fit_transform(X_train[:, i])
    X_train[:, 7] = Labelencoder_x.fit_transform(X_train[:, 7])
X_train
Labelencoder_y = LabelEncoder()
y_train = Labelencoder_y.fit_transform(y_train)
y_train
for i in range (0,5):
  X_test[:,i]= Labelencoder_x.fit_transform(X_test[:,i])
  X_test[:,7]= Labelencoder_x.fit_transform(X_test[:,7])
X_test
Labelencoder_y = LabelEncoder()
y_test = Labelencoder_y.fit_transform(y_test)
y_test
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
from sklearn import metrics
y_prediction = rf_clf.predict(X_test)
print("Accuracy of random forest classifier is ", metrics.accuracy_score(y_prediction,y_test))
y_prediction
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_prediction = nb_clf.predict(X_test)
print("Accuracy of bayes classifier is: ",metrics.accuracy_score(y_prediction,y_test))