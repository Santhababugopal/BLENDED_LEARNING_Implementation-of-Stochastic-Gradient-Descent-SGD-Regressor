# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries:

2.Import required libraries such as pandas, numpy, matplotlib, sklearn for the implementation.
Load the Dataset:

3.Load the dataset (e.g., CarPrice_Assignment.csv) using pandas.
Data Preprocessing:

4.Drop unnecessary columns (e.g., 'CarName', 'car_ID').
Handle categorical variables using pd.get_dummies().
Split the Data:

5.Split the dataset into features (X) and target variable (Y).
Split the data into training and testing sets using train_test_split().
Standardize the Data:

6.Standardize the feature data (X) and target variable (Y) using StandardScaler() to ensure they have mean=0 and variance=1.
Create the SGD Regressor Model:

7.Initialize the SGD Regressor model with max_iter=1000 and tol=1e-3.
Train the Model:

8.Fit the model to the training data using the fit() method.
Make Predictions:

9.Use the trained model to predict the target values for the test set.
Evaluate the Model:

10.Calculate performance metrics like Mean Squared Error (MSE) and R-squared score using mean_squared_error() and r2_score().
Display Model Coefficients:

11.Display the model's coefficients and intercept.
Visualize the Results:

12.Create a scatter plot comparing actual vs predicted prices.
End:

The program finishes by displaying the evaluation metrics, model coefficients, and a visual representation of the predictions.
Program:

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```
```
data = pd.read_csv("/content/CarPrice_Assignment (1) (1).csv")
print(data.head())
print(data.info())
```

```
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

```

```
X = data.drop('price', axis=1)
y = data['price']

```

```
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

```

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


```
```
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

```
```
sgd_model.fit(X_train, y_train)

```

```
y_pred = sgd_model.predict(X_test)

```

```

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

```
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

```

```
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept: ", sgd_model.intercept_)
```

```
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red') # Perfect prediction line
plt.show()

```
## Output:
LOAD THE DATASET

<img width="991" height="641" alt="image" src="https://github.com/user-attachments/assets/3576ecb1-b537-4f2f-9e86-510369dc96b9" />
<img width="738" height="729" alt="image" src="https://github.com/user-attachments/assets/26f4930f-20c0-4d74-800d-227bd96dd874" />


EVALUATION METRICS AND MODEL COEFFICIENTS


<img width="732" height="131" alt="image" src="https://github.com/user-attachments/assets/356f8eb7-a483-41f1-adb4-d2c96bc5f946" />
<img width="1050" height="268" alt="image" src="https://github.com/user-attachments/assets/ea6ae3d5-99dc-4509-b3df-d090a522df4e" />


VISUALIZATION OF ACTUAL VS PREDICTED VALUES:


<img width="1038" height="615" alt="image" src="https://github.com/user-attachments/assets/f34e7a0e-5182-448c-bf28-19c1669fc78c" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
