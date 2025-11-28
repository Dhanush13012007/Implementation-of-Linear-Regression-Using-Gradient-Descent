# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.
2.Scale the features using a standard scaler to normalize the data.
3.Initialize model parameters (theta) and add an intercept term to the feature set.
4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.
5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: m.dhanush
RegisterNumber:25009955

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression (X1, y):
    X = np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    num_iters = 1000   # number of iterations
    learning_rate = 0.01
    for _ in range(num_iters):
       predictions = X.dot(theta).reshape(-1, 1)
       errors = (predictions - y).reshape(-1, 1)
       theta = theta - learning_rate * (1 / len(X)) * X.T.dot(errors)
       return theta

data=pd.read_csv("/content/drive/MyDrive/50_Startups.csv")
data.head(11)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)


print("X =",X)

print("X1_Scaled =",X1_Scaled)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform (prediction)
print("prediction =",prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
![linear regression using gradient descent](sam.png)

<img width="678" height="429" alt="Screenshot 2025-11-28 143353" src="https://github.com/user-attachments/assets/ed75017e-a463-411d-b80b-ca904bcf95cd" />

<img width="398" height="726" alt="Screenshot 2025-11-28 143411" src="https://github.com/user-attachments/assets/a09450d0-8bc5-481b-bcfb-92f678f5c7df" />

<img width="632" height="707" alt="Screenshot 2025-11-28 143429" src="https://github.com/user-attachments/assets/029b4d90-ce8b-4484-8757-813669c25d6d" />


<img width="385" height="58" alt="Screenshot 2025-11-28 143440" src="https://github.com/user-attachments/assets/f87ae35e-f9e6-4061-96b8-f0007381b845" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
