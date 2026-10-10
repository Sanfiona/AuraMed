# AuraMed — System Rekomendacji Terapii Medycznych

Aplikacja webowa wspomagająca personalizację terapii medycznych na podstawie objawów pacjenta, zbudowana w technologii Streamlit z bazą danych PostgreSQL, zaawansowanymi algorytmami uczenia maszynowego oraz modułem badawczym (eksperymenty z szumem danych, metryki, krzywe ROC-AUC).

## Struktura projektu

```text
implementacja/
├── app.py                  # Główna aplikacja Streamlit (GUI)
├── train_model.py          # Trenowanie i ewaluacja 6 algorytmów ML
├── translations.py         # Słownik tłumaczeń PL/EN
├── requirements.txt        # Wymagane biblioteki Python
├── docker-compose.yml      # Konfiguracja Docker Compose
├── Dockerfile              # Konfiguracja obrazu aplikacji
├── dane/                   # Pliki CSV ze zbiorami danych
│   ├── Training.csv
│   ├── symtoms_df.csv
│   ├── Symptom-severity.csv
│   ├── medications.csv
│   ├── diets.csv
│   ├── precautions_df.csv
│   ├── workout_df.csv
│   └── description.csv
└── baza_danych/            # Skrypty bazy danych i migracje
    ├── schemat bazy danych drawio
    ├── baza_danych.sql
    └── skrypty_migracji/
        ├── import_training.py
        ├── import_precautions.py
        ├── import_diets.py
        ├── import_medications.py
        └── import_symptoms.py
```

## Wymagania wstępne

- Python 3.11+
- PostgreSQL 15+
- Docker i Docker Compose (dla wersji kontenerowej)

---

## Uruchomienie BEZ Dockera

### Krok 1 — Przejdź do folderu implementacji i zainstaluj biblioteki

```bash
cd implementacja
pip install -r requirements.txt
```

### Krok 2 — Konfiguracja bazy danych

Uruchom lokalną instancję PostgreSQL, utwórz bazę danych:

```sql
CREATE DATABASE AuraMed3;
```

Następnie zaimportuj główny schemat bazy:

```bash
psql -U postgres -d AuraMed3 -f baza_danych/baza_danych.sql
```

### Krok 3 — Pobierz dane z Kaggle

Pobierz zbiór danych z: https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset i umieść pliki CSV w folderze `implementacja/dane/`.

### Krok 4 — Import danych do bazy

Uruchom skrypty migracji wewnątrz folderu `implementacja`:

```bash
python baza_danych/skrypty_migracji/import_medications.py
python baza_danych/skrypty_migracji/import_symptoms.py
python baza_danych/skrypty_migracji/import_training.py
python baza_danych/skrypty_migracji/import_precautions.py
python baza_danych/skrypty_migracji/import_diets.py
```

### Krok 5 — Trenowanie modeli ML

```bash
python train_model.py
```

Skrypt wygeneruje pliki modeli (`.pkl`) w katalogu projektu.

### Krok 6 — Uruchomienie aplikacji

```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem: `http://localhost:8501`

---

## Uruchomienie Z Dockerem 

### Krok 1 — Przygotowanie danych
Upewnij się, że pliki CSV znajdują się w folderze `implementacja/dane/`.

### Krok 2 — Uruchomienie kontenerów
Przejdź do folderu `implementacja` i uruchom środowisko:

```bash
cd implementacja
docker-compose up -d --build
```

Docker automatycznie:
- Uruchomi kontener z bazą PostgreSQL i zainicjalizuje schemat z pliku `baza_danych.sql`.
- Uruchomi kontener setup, który zaimportuje dane oraz wytrenuje modele.
- Uruchomi aplikację Streamlit.

### Krok 3 — Dostęp do aplikacji
Aplikacja jest dostępna pod adresem: `http://localhost:8501`

---

## Zakładki aplikacji

| Zakładka | Opis |
|---|---|
| **Diagnostyka** | Wprowadzanie objawów pacjenta i generowanie rekomendacji terapii |
| **Statystyki** | Eksploracyjna analiza danych i statystyki medyczne |
| **Porównanie algorytmów** | Porównanie 6 algorytmów ML, macierze błędu, krzywe ROC-AUC oraz eksperymenty z odpornością na szum danych (5%, 10%, 20%) |
| **Historia** | Przeglądanie historii zapytań i rekomendacji z bazy danych |

---

## Konfiguracja połączenia z bazą

Domyślny ciąg połączenia (zmień w razie potrzeby w kodzie źródłowym):

```python
DB_URL = 'postgresql://postgres:postgres@db:5432/AuraMed3' # W Dockerze
DB_URL = 'postgresql://postgres:postgres@localhost:5432/AuraMed3' # Lokalnie
```

---

## Uwagi

- System wspiera podejmowanie decyzji klinicznych, jednak ostateczna diagnoza zawsze należy do personelu medycznego.