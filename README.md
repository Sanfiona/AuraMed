# AuraMed — System Rekomendacji Terapii Medycznych

Aplikacja webowa wspomagająca personalizację terapii medycznych na podstawie objawów pacjenta, zbudowana w technologii Streamlit z bazą danych PostgreSQL i algorytmami uczenia maszynowego.

## Struktura projektu

```
implementacja/
├── app.py                  # Główna aplikacja Streamlit
├── train_model.py          # Trenowanie i eksport modeli ML
├── translations.py         # Słownik tłumaczeń PL/EN
├── requirements.txt        # Wymagane biblioteki Python
dane/
baza_danych/
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
- Docker i Docker Compose (opcjonalnie)

---

## Uruchomienie BEZ Dockera

### Krok 1 — Instalacja bibliotek

```bash
pip install -r requirements.txt
```

### Krok 2 — Konfiguracja bazy danych

Uruchom PostgreSQL i utwórz bazę danych:

```sql
CREATE DATABASE AuraMed3;
```

Następnie wykonaj schemat bazy:

```bash
psql -U postgres -d AuraMed3 -f init/schema.sql
```

### Krok 3 — Pobierz dane z Kaggle

Pobierz zbiór danych z: https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset

Umieść pliki CSV w folderze `dane/`:
```
dane/
├── Training.csv
├── symtoms_df.csv
├── Symptom-severity.csv
├── medications.csv
├── diets.csv
├── precautions_df.csv
├── workout_df.csv
└── description.csv
```

### Krok 4 — Import danych do bazy

Uruchom skrypty migracji w kolejności:

```bash
python skrypty_migracji/import_medications.py  # tworzy choroby i leki
python skrypty_migracji/import_symptoms.py     # tworzy objawy z wagami
python skrypty_migracji/import_training.py     # dane treningowe
python skrypty_migracji/import_precautions.py  # zalecenia (precautions)
python skrypty_migracji/import_diets.py        # diety
```

### Krok 5 — Trenowanie modeli ML

```bash
python train_model.py
```

Powstanie 5 plików `.pkl` w folderze `implementacja/`.

### Krok 6 — Uruchomienie aplikacji

```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem: http://localhost:8501

---

## Zakładki aplikacji

| Zakładka | Opis |
|---|---|
| Diagnostyka | Wprowadzanie objawów i generowanie rekomendacji |
| Statystyki | Wykresy eksploracyjne danych |
| Porównanie algorytmów | Interaktywny eksperyment z szumem |
| Historia | Przeglądanie historii rekomendacji |

---

## Konfiguracja połączenia z bazą

Domyślne ustawienia w `app.py` i skryptach migracji:

```python
DB_URL = 'postgresql://postgres:postgres@localhost:5432/AuraMed3'
```

---

## Uwagi

- System służy wyłącznie jako wsparcie decyzji klinicznych
- Ostateczna diagnoza zawsze należy do lekarza
- Modele `.pkl` muszą być wygenerowane przed uruchomieniem aplikacji