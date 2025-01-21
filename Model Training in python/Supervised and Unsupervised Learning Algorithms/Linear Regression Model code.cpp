import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
dataset=pd.read_csv("Housing.csv")
x=dataset.drop(["price"],axis=1)
y=dataset['price']
bool_col = ["furnishingstatus",'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in bool_col:
    x[col]=x[col].map({'yes':1,'no':0}).fillna(0)
    x["furnishingstatus"]=x["furnishingstatus"].map({'furnished':1,'unfurnished':0,'semi-furnished':0.5}).fillna(0)
    x_train , x_test , y_train , y_test =  train_test_split(x , y , test_size=0.3 , random_state=42)
lr= LinearRegression()
lr.fit(x_train , y_train)
ypred=lr.predict(x_test)
mse=mean_squared_error(ypred,y_test)
mae=mean_absolute_error(ypred,y_test)
r2=r2_score(ypred,y_test)
mse=mean_squared_error(ypred,y_test)
mae=mean_absolute_error(ypred,y_test)
r2=r2_score(ypred,y_test)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
