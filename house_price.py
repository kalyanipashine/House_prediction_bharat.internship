import pandas as pd
import numpy as np

data=pd.read_csv('ParisHousing.csv')
print(data.head)
print(data.shape)
print(data.info)


for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)

print(data.isna().sum())
print(data.describe)

data['numberOfRooms'].value_counts()
print(data.head)

data['price_per_sqft'] = data['price'] * 100000 / data['squareMeters']
print(data['price_per_sqft'])
print(data.describe)
print(data.shape)
print(data)

data.drop(columns=['price_per_sqft'],inplace=True)
print(data.head)

data.to_csv("final_dataset.csv")

X=data.drop(columns=['price'])
y=data['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)

column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['numberOfRooms']), remainder='passthrough')
scaler = StandardScaler()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# X being feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LinearRegression()
lr.fit(X_scaled, y)

pipe = make_pipeline(column_trans,scaler, lr)

pipe.fit(X_train,y_train)

y_pred_lr = pipe.predict(X_test)

r2_score(y_test,y_pred_lr)

lasso = Lasso()
pipe = make_pipeline(column_trans,scaler, lasso)
pipe.fit(X_train,y_train)

y_pred_lasso = pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)

ridge = Ridge()
pipe = make_pipeline(column_trans,scaler, ridge)
pipe.fit(X_train,y_train)

y_pred_ridge = pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)

print("No Regularization: ", r2_score(y_test,y_pred_lr))
print("Lasso: ", r2_score(y_test,y_pred_lasso))
print("Ridge: ", r2_score(y_test,y_pred_ridge))

import pickle
pickle.dump(pipe, open('RidgeModel.pkl','wb'))
