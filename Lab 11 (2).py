#!/usr/bin/env python
# coding: utf-8

# # Task 1

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd . read_csv (r"Salary.csv")
print(dataset)
x = dataset ['YearsExperience'].values.reshape (-1 ,1)
y = dataset ['Salary'].values.reshape ( -1 ,1)
dataset.plot(x='YearsExperience', y='Salary', style='o')
plt.title('YearsExperience vs Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)

yPrediction = linearRegressor.predict(xTest)

df = pd.DataFrame({'Actual': yTest.flatten(), 'Predicted': yPrediction.flatten()})
print(df)

plt.scatter(xTrain, yTrain, color='red')
plt.plot(xTrain, linearRegressor.predict(xTrain), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(xTest, yTest, color='red')
plt.plot(xTest, yPrediction, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(yTest, yPrediction))
print('Mean Squared Error:', metrics.mean_squared_error(yTest, yPrediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yTest, yPrediction)))


# # Task 2

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_excel(r"weather.xlsx")

dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.fillna(dataset.mean())

X_gdd = dataset[['Basel Growing Degree Days [2 m elevation corrected]']]
y_gdd = dataset['Basel Temperature [2 m elevation corrected]']

X_train_gdd, X_test_gdd, y_train_gdd, y_test_gdd = train_test_split(X_gdd, y_gdd, test_size=0.2, random_state=0)

linear_regressor_gdd = LinearRegression()
linear_regressor_gdd.fit(X_train_gdd, y_train_gdd)

y_pred_gdd = linear_regressor_gdd.predict(X_test_gdd)

mae_gdd = metrics.mean_absolute_error(y_test_gdd, y_pred_gdd)
mse_gdd = metrics.mean_squared_error(y_test_gdd, y_pred_gdd)
rmse_gdd = np.sqrt(mse_gdd)

print('Basel Growing Degree Days')
print('Mean Absolute Error:', mae_gdd)
print('Mean Squared Error:', mse_gdd)
print('Root Mean Squared Error:', rmse_gdd)
print()

X_ws = dataset[['Basel Wind Speed [10 m]']]
y_ws = dataset['Basel Temperature [2 m elevation corrected]']

X_train_ws, X_test_ws, y_train_ws, y_test_ws = train_test_split(X_ws, y_ws, test_size=0.2, random_state=0)

linear_regressor_ws = LinearRegression()
linear_regressor_ws.fit(X_train_ws, y_train_ws)

y_pred_ws = linear_regressor_ws.predict(X_test_ws)

mae_ws = metrics.mean_absolute_error(y_test_ws, y_pred_ws)
mse_ws = metrics.mean_squared_error(y_test_ws, y_pred_ws)
rmse_ws = np.sqrt(mse_ws)

print('Basel Wind Speed')
print('Mean Absolute Error:', mae_ws)
print('Mean Squared Error:', mse_ws)
print('Root Mean Squared Error:', rmse_ws)
print()

X_wd = dataset[['Basel Wind Direction [10 m]']]
y_wd = dataset['Basel Temperature [2 m elevation corrected]']

X_train_wd, X_test_wd, y_train_wd, y_test_wd = train_test_split(X_wd, y_wd, test_size=0.2, random_state=0)

linear_regressor_wd = LinearRegression()
linear_regressor_wd.fit(X_train_wd, y_train_wd)

y_pred_wd = linear_regressor_wd.predict(X_test_wd)

mae_wd = metrics.mean_absolute_error(y_test_wd, y_pred_wd)
mse_wd = metrics.mean_squared_error(y_test_wd, y_pred_wd)
rmse_wd = np.sqrt(mse_wd)

print('Basel Wind Direction')
print('Mean Absolute Error:', mae_wd)
print('Mean Squared Error:', mse_wd)
print('Root Mean Squared Error:', rmse_wd)


# # Task 3

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)

df = california_housing.data
df['PRICE'] = california_housing.target


corr_matrix = df.corr()
selected_features = corr_matrix['PRICE'].abs().sort_values(ascending=False)[:5].index
X = df[selected_features]
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
model = GridSearchCV(LinearRegression(), param_grid, cv=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared (R2) Score:', r2)
print('Best Parameters:', model.best_params_)

