W celu uruchomienia projektu należy zainstalować następujące biblioteki:
`pip install pandas scikit-learn matplotlib seaborn`

Projekt składa się z dwóch plików:
- classifier_generic.py - generyczna wersja programu, pozwala załadować zbiór danych
w postaci pliku CSV, oraz podać prawdopodobieństwa a priori. Generuje tabelę wyników, dokładność i macierz pomyłek.
Format pliku CSV zakłada, że ostatnia kolumna zawiera klasy, a poprzednie - cechy. Pierwszy wiersz zawiera nagłówki.
- classifier.py - wersja dostosowana do zbioru Customer_Behaviour.csv w celu wizualizacji. Dodatkowo
generuje grafy wizualizujące obszary decyzyjne, jak i wizualną reprezentację macierzy pomyłek.
- do obu plików dodane są komentarze w celu objaśnienia kodu