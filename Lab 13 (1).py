#!/usr/bin/env python
# coding: utf-8

# # Task 1

# In[16]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


data = pd.read_excel("Temp.xlsx")


X = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values


encoder = LabelEncoder()
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 1] = encoder.fit_transform(X[:, 1])
y = encoder.fit_transform(y)


classifier = GaussianNB()


classifier.fit(X, y)


prediction = classifier.predict([[0, 1]])

print("Predicted Value:", prediction)


# # Task 2

# In[17]:



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv('Dermatology data set.csv')


features = data.iloc[:, :-1]
target = data.iloc[:, -1]


features = features.fillna(features.mean())


train_size = 0.7
test_size = 0.3


X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=train_size, test_size=test_size, random_state=42)


classifier = GaussianNB()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


k = 5  
scores = cross_val_score(classifier, features, target, cv=k)
avg_accuracy = np.mean(scores)

print(f"\nCross-Validation (k={k})")
print(f"Average Accuracy: {avg_accuracy:.3f}")



# # Task 3

# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv('Bank authentication data set.csv')


features = data.iloc[:, :-1]
target = data.iloc[:, -1]


features = features.fillna(features.mean())


train_size = 0.7
test_size = 0.3


X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=train_size, test_size=test_size, random_state=42)


nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)


y_pred = nb_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


k = 5 
scores = cross_val_score(nb_classifier, features, target, cv=k)
avg_accuracy = np.mean(scores)

print(f"\nCross-Validation (k={k})")
print(f"Average Accuracy: {avg_accuracy:.3f}")


# # Task 4

# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


data = pd.read_csv('Dermatology data set.csv')


features = data.iloc[:, :-1]
target = data.iloc[:, -1]


features = features.fillna(features.mean())


features_train, features_test, target_train, target_test = train_test_split(features, target, train_size=0.7, test_size=0.3, random_state=42)


nb_model = GaussianNB()
nb_model.fit(features_train, target_train)


nb_predictions = nb_model.predict(features_test)
nb_accuracy = accuracy_score(target_test, nb_predictions)
nb_f1 = f1_score(target_test, nb_predictions, average='weighted')

print("Naïve Bayes Results:")
print(f"Accuracy: {nb_accuracy:.3f}")
print(f"F1-score: {nb_f1:.3f}")


svm_model = SVC()
svm_model.fit(features_train, target_train)


svm_predictions = svm_model.predict(features_test)
svm_accuracy = accuracy_score(target_test, svm_predictions)
svm_f1 = f1_score(target_test, svm_predictions, average='weighted')

print("\nSupport Vector Machine (SVM) Results:")
print(f"Accuracy: {svm_accuracy:.3f}")
print(f"F1-score: {svm_f1:.3f}")


k = 10  


nb_scores = cross_val_score(nb_model, features, target, cv=k, scoring='f1_weighted')
nb_avg_f1 = np.mean(nb_scores)
nb_avg_accuracy = np.mean(cross_val_score(nb_model, features, target, cv=k))


svm_scores = cross_val_score(svm_model, features, target, cv=k, scoring='f1_weighted')
svm_avg_f1 = np.mean(svm_scores)
svm_avg_accuracy = np.mean(cross_val_score(svm_model, features, target, cv=k))

print("\nCross-Validation Results (10-fold):")
print("Naïve Bayes")
print(f"Average Accuracy: {nb_avg_accuracy:.3f}")
print(f"Average F1-score: {nb_avg_f1:.3f}")

print("\nSupport Vector Machine (SVM)")
print(f"Average Accuracy: {svm_avg_accuracy:.3f}")
print(f"Average F1-score: {svm_avg_f1:.3f}")


# In[ ]:




