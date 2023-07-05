#!/usr/bin/env python
# coding: utf-8

# # Task 1 and 2

# In[2]:


import numpy as np

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1

for _ in range(10000):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1, True)
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)


# # Task 3

# In[28]:


import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Read the data from the Excel file
data = pd.read_excel("weather.xlsx")

# Extract the timestamp column
timestamp = data["timestamp"]

# Remove the timestamp column from the original data
data = data.drop("timestamp", axis=1)

# Handle missing values by replacing them with the mean
data.fillna(data.mean(), inplace=True)

# Check for infinite or extremely large values
if np.any(np.isinf(data.values)) or np.any(np.abs(data.values) > 1e6):
    # Handle infinite or large values as per your dataset requirements
    data = data.replace([np.inf, -np.inf], np.nan)
    data.fillna(data.mean(), inplace=True)

# Create the DataFrame X without the timestamp column
X = data.drop(columns=["Basel Temperature [2 m elevation corrected]"])

# Set the target variable y as the Basel Temperature column
y = data["Basel Temperature [2 m elevation corrected]"]

# Convert timestamp to datetime format
timestamp = pd.to_datetime(timestamp, format="%Y%m%dT%H%M")

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the neural network model
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:




