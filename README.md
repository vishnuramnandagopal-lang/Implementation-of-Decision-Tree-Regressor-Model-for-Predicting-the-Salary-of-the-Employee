# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
## Program:
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: vishnuram g n
RegisterNumber:  212225240187
*/
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv("Salary_EX7.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Position"] = le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

plt.figure(figsize=(20, 8))

plot_tree(dt, feature_names=x.columns, filled=True)

plt.show()


## Output:

<img width="411" height="222" alt="Screenshot 2026-03-18 153605" src="https://github.com/user-attachments/assets/60ff01e8-2c03-4a4a-9361-762c1163bb6d" />

<img width="1220" height="483" alt="Screenshot 2026-03-18 153618" src="https://github.com/user-attachments/assets/e6dcb649-c32d-47e8-94e4-bbf70b69a31a" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
