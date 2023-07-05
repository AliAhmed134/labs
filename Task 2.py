import pandas as pd
from sklearn.preprocessing import StandardScaler


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data = pd.read_csv(url, names=names)


print(iris_data.head())


encoded_data = pd.get_dummies(iris_data, columns=["class"])


scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data.iloc[:, :-3])  
scaled_df = pd.DataFrame(scaled_data, columns=encoded_data.columns[:-3])

print(scaled_df.head())
