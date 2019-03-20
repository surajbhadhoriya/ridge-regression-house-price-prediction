

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#POLYNOMIAL RIDGE  REGRESSION
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#loas Data
data1=pd.read_csv('C:/Users/SURAJ BHADHORIYA/Desktop/SINGLE-FILES/kc_house_data.csv')
data1.info()
x=data1['sqft_living']
y=data1['price']
x1=sorted(x,key=None,reverse=False)
y1=sorted(y,key=None,reverse=False)
x1=np.array(x1,dtype=np.int64)
y1=np.array(y1,dtype=np.int64)
#Apply polynomial feature with degree=15
from sklearn.preprocessing import PolynomialFeatures
pol_reg= PolynomialFeatures(degree=10)
x1=x1.reshape(-1,1)
x_pol=pol_reg.fit_transform(x1)
#split data 
X_train, X_test, y_train, y_test=train_test_split(x_pol, y1, test_size=0.2, random_state=5)
#apply model to polynomial regression
model2=LinearRegression()
model2.fit(X_train, y_train)
b1=model2.intercept_
m1=model2.coef_
print("inertcept",b1)
print("slope",m1)
ac1=model2.score(X_test,y_test)
print("accuracy",ac1)
y_pred1=model2.predict(X_test)
print("prediction",y_pred1)
#VISIULIZATION
plt.scatter(x1,y1,color='red')
plt.plot(x1,model2.predict(pol_reg.fit_transform(x1)),color='blue')
plt.tittle("Truth or bbluff (linear regression)")
plt.xlabel("squarfit_living")
plt.ylabel("price")
plt.show()
#-------------Above model is overfitted--------------------

#to avoid over fitting ridge regression require
#apply ridge regression
from sklearn.linear_model import Ridge
ridmodel=Ridge(alpha=0.000000000000005,normalize=True)
ridmodel.fit(X_train,y_train)
rid_pre=ridmodel.predict(X_test)
print(rid_pre)
ac2=ridmodel.score(X_test,y_test)
print("accuracy",ac2)
#Data visiulization
plt.scatter(x1,y1,color='red')
plt.plot(x1,ridmodel.predict(pol_reg.fit_transform(x1)),color='blue')
plt.tittle("Truth or bbluff (linear regression)")
plt.xlabel("squarfit_living")
plt.ylabel("price")
plt.show()


