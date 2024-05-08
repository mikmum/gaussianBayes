import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ładowanie danych
data = pd.read_csv('Customer_Behaviour.csv')

# Przetwarzanie danych
# Zamiana "Gender" na wartości numeryczne
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Podział na cechy (X) i etykiety (y)
X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizacja cech (opcjonalna, ale często pomaga w stabilizacji)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Klasyfikator naiwny Bayes
model = GaussianNB()

# Trenowanie modelu
model.fit(X_train_scaled, y_train)

# Prognozowanie
y_pred = model.predict(X_test_scaled)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Wizualizacja obszarów decyzyjnych dla dwóch cech (Age i EstimatedSalary)
# Tworzymy siatkę wartości
age_min, age_max = X['Age'].min() - 1, X['Age'].max() + 1
salary_min, salary_max = X['EstimatedSalary'].min() - 1000, X['EstimatedSalary'].max() + 1000

xx, yy = np.meshgrid(np.arange(age_min, age_max, 0.1),
                     np.arange(salary_min, salary_max, 1000))
Z = model.predict(scaler.transform(np.c_[np.zeros_like(xx.ravel()), xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdYlBu)
plt.scatter(X_train['Age'], X_train['EstimatedSalary'], c=y_train, edgecolors='k', marker='o', s=100, label='Train')
plt.scatter(X_test['Age'], X_test['EstimatedSalary'], c=y_pred, edgecolors='k', marker='x', s=100, label='Test')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Obszary decyzyjne z wykorzystaniem klasyfikatora bayesowskiego')
plt.legend()
plt.show()
