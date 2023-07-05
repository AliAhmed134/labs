import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()


X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_neighbors = [1, 3, 5, 7, 9]


for n in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)


    knn.fit(X_train, y_train)


    y_pred = knn.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)


    print(f"Number of Neighbors: {n}, Accuracy: {accuracy}")
