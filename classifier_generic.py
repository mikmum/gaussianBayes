import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix


# Generyczna wersja programu. Zczytuje plik csv i prawdopodobieństwa a priori, następnie przetwarza etykiety i tworzy
# predykcje. Założenie odnośnie pliku csv jest takie, że ostatnia kolumna zawiera etykiety
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


# Wprowadzenie danych z CSV oraz opcjonalnych a priori prawdopodobieństw
input_data = input("Podaj nazwę pliku CSV oraz opcjonalne prawdopodobieństwa a priori (rozdzielone spacją): ")
input_parts = input_data.split()

# Załadowanie danych
filename = input_parts[0]
data = pd.read_csv(filename)

# Czytanie prawdopodobieństw a priori (jeśli są podane)
priors = None
if len(input_parts) > 1:
    try:
        priors = [float(p) for p in input_parts[1:]]  # Konwersja na float
    except ValueError:
        print("Nieprawidłowe wartości prawdopodobieństw a priori. Użycie domyślnych.")

# Przygotowanie kolumn etykiet do konwersji
label_encoders = {}
class_names = None

# Sprawdzanie i konwersja kolumn tekstowych do numerycznych, jeśli wymagane
for col in data.columns:
    if data[col].dtype == 'O':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        # Jeśli to ostatnia kolumna (etykieta), zapisz nazwy klas - pomaga to potem w tworzeniu czytelniejszej tabeli
        if col == data.columns[-1]:
            class_names = le.classes_

# Zakładamy, że ostatnia kolumna to etykieta (klasyfikacja), reszta to cechy
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Normalizacja cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Użycie własnego klasyfikatora z opcjonalnymi a priori prawdopodobieństwami
model = GNBClassifier(priors=priors)

# Trenowanie modelu
model.fit(X_train_scaled, y_train)

# Prognozowanie
y_pred = model.predict(X_test_scaled)

# Prognozowanie prawdopodobieństw a posteriori
y_proba = model.predict_proba(X_test_scaled)

# Utworzenie słownika z prawdopodobieństwami dla każdej kategorii
if class_names is None:  # Jeśli nie użyto LabelEncoder dla ostatniej kolumny
    class_names = [str(i) for i in range(y_proba.shape[1])]

proba_dict = {
    f"Prawdopodobieństwo dla etykiety {class_names[i]}": y_proba[:, i]
    for i in range(y_proba.shape[1])  # Dynamiczne tworzenie kolumn na podstawie liczby klas
}

# Utworzenie tabeli z informacjami o próbkach i prawdopodobieństwami dla każdej kategorii
results_df = pd.DataFrame({
    "Nr_próbki": np.arange(1, len(y_test) + 1),
    **{col: X_test[col].values for col in X_test.columns},
    "Kategoria": y_test,
    "Predykcja": y_pred,
    **proba_dict  # Dodanie wszystkich kolumn z prawdopodobieństwami
})

# Wydrukowanie tabeli w terminalu
print("Tabela wyników:")
print(results_df.to_string(index=False))  # Wyświetlenie tabeli bez indeksów

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Macierz Pomyłek:")
print(conf_matrix)
