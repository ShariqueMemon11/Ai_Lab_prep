import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
dataset=pd.read_csv("Housing.csv")
x=dataset.drop(["price","furnishingstatus",'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'],axis=1)
y=dataset["price"]
x_train , x_test , y_train , y_test =  train_test_split(x , y , test_size=0.3 , random_state=42)
k_values=list(range(1,21))
acc_val=[]
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train , y_train)
    y_pred=knn.predict(x_test)
    acc=accuracy_score(y_pred , y_test)
    acc_val.append(acc)
plt.figure(figsize=(10, 6))
plt.plot(k_values, acc_val, marker='o', linestyle='-')
plt.title('Accuracy vs. Number of Neighbors (k) for KNN')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(k_values)
plt.show()
