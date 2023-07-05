import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
column_names = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
                "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
                "knee_and_elbow_involvement", "scalp_involvement", "family_history",
                "melanin_incontinence", "eosinophils_infiltrate", "PNL_infiltrate",
                "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis",
                "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
                "elongation_of_the_rete_ridges", "thinning_of_the_suprapapillary_epidermis",
                "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
                "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer",
                "spongiosis", "saw-tooth_appearance_of_retes", "follicular_horn_plug",
                "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate",
                "band-like_infiltrate", "age", "class"]
dermatology_data = pd.read_csv(url, names=column_names)


dermatology_data.replace('?', float('nan'), inplace=True)


dermatology_data.fillna(dermatology_data.mode().iloc[0], inplace=True)


X = dermatology_data.iloc[:, :-1]
y = dermatology_data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


cv_scores = cross_val_score(knn, X, y, cv=10)
avg_accuracy = cv_scores.mean()
print("Average Accuracy (10-Fold Cross Validation):", avg_accuracy)
