import pandas as pd
from sklearn.preprocessing import StandardScaler


data = pd.DataFrame({
    'Job Id': [334, 234, 138, 463, 283, 88, 396, 470, 335, 272, 237, 318, 84, 311, 163, 453, 176, 449, 11],
    'Burst time': [179, 340, 143, 264, 216, 36, 128, 203, 271, 399, None, 311, 111, 87, 103, 213, 251, 49, 168],
    'Arrival Time': [0.6875, 0.78, 0.915, None, 0.555, 0.6625, 0.1975, 0.9875, 0.0275, 0.215, 0.4825, 0.5675, 0.2725, None, 0.46, 0.0775, 0.705, 0.255, 0.3175],
    'Prremptive': [1, 0, 1, 0, 0, 0, 1, 1, 0, None, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    'Resources': [4, 4, 4, 5, 6, 5, None, 4, 3, 3, 4, 1, 2, 7, 5, 1, 6, 4, 7]
})


data['Burst time'].fillna(data['Burst time'].mean(), inplace=True)
data['Arrival Time'].fillna(data['Arrival Time'].median(), inplace=True)
data['Prremptive'].fillna(data['Prremptive'].mode()[0], inplace=True)
data['Resources'].fillna(data['Resources'].mode()[0], inplace=True)


scaler = StandardScaler()
data[['Burst time', 'Arrival Time']] = scaler.fit_transform(data[['Burst time', 'Arrival Time']])

print(data)
