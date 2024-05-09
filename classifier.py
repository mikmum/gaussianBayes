import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Ten program jest dostosowany pod zbiór Customer_Behaviour.csv, by umożliwić wizualizację.
# Klasa GNBClassifier jest identyczna jak w generycznej wersji.
# Klasyfikacje przeprowadzamy według danych (wiek, zarobki)
class GNBClassifier:
    def __init__(self, priors=None):
        self.priors = priors
        self.means_ = None
        self.variances_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Unikalne etykiety
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))

        # Obliczanie średnich i wariancji dla każdej etykiety
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]  # Próbki danej etykiety
            self.means_[i] = np.mean(X_c, axis=0)  # Średnia dla cechy
            self.variances_[i] = np.var(X_c, axis=0)  # Wariancja dla cechy

        # Obliczanie prawdopodobieństw a priori jeśli nie było podane wprost
        if self.priors is None:
            self.priors = np.array([np.mean(y == c) for c in self.classes_])

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        likelihood = np.zeros((n_samples, n_classes))

        # Obliczanie prawdopodobieństw warunkowych dla każdej etykiety
        for i in range(n_classes):
            for j in range(X.shape[1]):
                likelihood[:, i] += norm.logpdf(
                    X[:, j], loc=self.means_[i, j], scale=np.sqrt(self.variances_[i, j])
                )

        # Prawdopodobieństwo a posteriori z twierdzenia Bayesa
        log_prior = np.log(self.priors)
        posterior = likelihood + log_prior  # Dodanie priorytetów
        # posterior jest normalizowany przez odejmowanie maksymalnej wartości dla stabilności numerycznej
        posterior = np.exp(posterior - posterior.max(axis=1, keepdims=True))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)  # Normalizacja

        return posterior

    def predict(self, X):
        # Wybieramy klasę z najwyższym prawdopodobieństwem
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# Wprowadzenie danych z CSV
data = pd.read_csv("Customer_Behaviour.csv")

# Podział na cechy (X) i etykiety (y)
X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"]

# Podział na zbiór treningowy i testowy. Okreslamy rozmiar testowego jako wartość od 0 do 1, reszta będzie zbiorem treningowym.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Klasyfikator z priorytetami. Zakładamy bez żadnych informacji że prawdopodobieństwo kupna
# to 0.4
priors = [0.5, 0.5]
model = GNBClassifier(priors)

# Trenowanie modelu
model.fit(X_train_scaled, y_train)

# Prognozowanie
y_pred = model.predict(X_test_scaled)

# Prognozowanie prawdopodobieństw a posteriori
y_proba = model.predict_proba(X_test_scaled)

# Wizualizacja obszarów decyzyjnych dla dwóch cech (Wiek i Zarobki)
age_min, age_max = X["Age"].min() - 1, X["Age"].max() + 1
salary_min, salary_max = X["EstimatedSalary"].min() - 1000, X["EstimatedSalary"].max() + 1000

xx, yy = np.meshgrid(
    np.arange(age_min, age_max, 0.1),
    np.arange(salary_min, salary_max, 1000)
)

viz_data = np.c_[xx.ravel(), yy.ravel()]  # Dodanie brakującej kolumny dla zgodności

Z = model.predict(scaler.transform(viz_data))
Z = Z.reshape(xx.shape)

# Pierwszy wykres - tylko dane ze zbioru treningowego (Wiek i Zarobki)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap="bwr_r")

colors_train = ["red" if val == 0 else "blue" for val in y_train]

plt.scatter(
    X_train["Age"],
    X_train["EstimatedSalary"],
    c=colors_train,
    edgecolors="k",
    marker="o",
    s=100,
    label="Train (0: Red, 1: Blue)"
)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Obszary decyzyjne - zbiór treningowy (Wiek i Zarobki)")
plt.legend(loc="upper right")
plt.show()

# Drugi wykres - wartości przewidywane ze zbioru testowego (Wiek i Zarobki)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap="bwr_r")

colors_test_pred = ["red" if val == 0 else "blue" for val in y_pred]

plt.scatter(
    X_test["Age"],
    X_test["EstimatedSalary"],
    c=colors_test_pred,
    edgecolors="k",
    marker="x",
    s=100,
    label="Test (przewidywane) (0: Red, 1: Blue)"
)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Obszary decyzyjne - wartości przewidywane (Wiek i Zarobki)")
plt.legend(loc="upper right")
plt.show()

# Trzeci wykres - faktyczne wartości ze zbioru testowego (Wiek i Zarobki)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap="bwr_r")

colors_test_actual = ["red" if val == 0 else "blue" for val in y_test]

plt.scatter(
    X_test["Age"],
    X_test["EstimatedSalary"],
    c=colors_test_actual,
    edgecolors="k",
    marker="x",
    s=100,
    label="Test (faktyczne) (0: Red, 1: Blue)"
)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Obszary decyzyjne - faktyczne wartości (Wiek i Zarobki)")
plt.legend(loc="upper right")
plt.show()

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Wizualizacja macierzy pomyłek
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Faktyczne: Nie", "Faktyczne: Tak"],
    columns=["Przewidywane: Nie", "Przewidywane: Tak"]
)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Macierz pomyłek")
plt.show()

# Utworzenie tabeli z informacjami o próbkach
results_df = pd.DataFrame({
    "Nr próbki": np.arange(1, len(y_test) + 1),
    "Wiek": X_test['Age'],
    "Pensja": X_test['EstimatedSalary'],
    "Kategoria": y_test,
    "Predykcja": y_pred,
    "Prawdopodobieństwo kupna": y_proba[:, 1],
})

print("Tabela wyników (Płeć i Wiek):")
print(results_df.to_string(index=False))  # Bez indeksów w wyświetlanej tabeli