# AuraMed — System Rekomendacji Terapii Medycznych

Aplikacja webowa wspomagająca personalizację terapii medycznych na podstawie objawów pacjenta, zbudowana w technologii Streamlit z bazą danych PostgreSQL, zaawansowanymi algorytmami uczenia maszynowego oraz modułem badawczym (eksperymenty z szumem danych, metryki, krzywe ROC-AUC).

## Wersja demonstracyjna (Online)

Aplikacja jest wdrożona na serwerze VPS i zabezpieczona szyfrowaniem HTTPS (Let's Encrypt via Caddy) oraz uwierzytelnianiem HTTP Basic Auth:

- **Adres URL:** `https://185-33-145-126.nip.io`
- **Login:** `recenzent`
- **Hasło:** `AuraMed2026!`

---

## Struktura projektu

```text
implementacja/
├── app.py                  # Główna aplikacja Streamlit (GUI)
├── train_model.py          # Trenowanie i ewaluacja algorytmów ML
├── translations.py         # Słownik tłumaczeń PL/EN
├── requirements.txt        # Wymagane biblioteki Python
├── docker-compose.yml      # Konfiguracja Docker Compose (Postgres, Streamlit, Caddy)
├── Dockerfile              # Konfiguracja obrazu aplikacji
├── Caddyfile               # Konfiguracja reverse proxy, HTTPS oraz Basic Auth
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
        ├── import_trening.py
        ├── import_precautions.py
        ├── import_diets.py
        ├── import_medications.py
        └── import_symptoms.py
```

## Wymagania wstępne

- Python 3.11+
- PostgreSQL 15+
- Docker i Docker Compose v2 (dla wersji kontenerowej)

---

## Uruchomienie Z Dockerem (Rekomendowane / Serwer VPS)

Architektura kontenerowa składa się z czterech usług zdefiniowanych w `docker-compose.yml`:
1. **`db`** — baza danych PostgreSQL 15 z wolumenem trwałym `postgres_data`.
2. **`setup`** — jednorazowy kontener inicjalizujący bazę danymi z plików CSV oraz trenujący modele ML (`.pkl`).
3. **`app`** — kontener z aplikacją webową Streamlit działający w izolowanej sieci wewnętrznej Dockera.
4. **`caddy`** — serwer WWW pełniącą rolę *reverse proxy*, automatycznie zarządzający certyfikatem TLS/HTTPS oraz wymuszający logowanie Basic Auth przed dostępem do interfejsu.

### Krok 1 — Przygotowanie danych i konfiguracji
Upewnij się, że pliki CSV znajdują się w folderze `implementacja/dane/`, a plik `Caddyfile` zawiera adres domeny oraz zahashowane hasło dostępowe (wygenerowane poleceniem `caddy hash-password`).

### Krok 2 — Inicjalizacja bazy danych i trenowanie modeli
Przejdź do folderu `implementacja` i uruchom jednorazowy kontener konfiguracyjny:

```bash
cd implementacja
docker-compose --profile setup up -d --build
docker compose run --rm setup
```

### Krok 3 — Uruchomienie środowiska produkcyjnego
Uruchom docelowe kontenery w tle:

```bash
docker compose up -d --build
```

Docker automatycznie:
- Uruchomi bazę PostgreSQL i zainicjalizuje schemat z pliku `baza_danych.sql`.
- Uruchomi aplikację Streamlit na wewnętrznym porcie `8501`.
- Uruchomi serwer Caddy nasłuchujący na portach `80` i `443`, kierujący ruch do aplikacji po poprawnym uwierzytelnieniu.

### Krok 4 — Dostęp do aplikacji
Aplikacja jest dostępna pod adresem: `https://185-33-145-126.nip.io` (lub `http://localhost` przy konfiguracji lokalnej).

---

## Uruchomienie BEZ Dockera (Lokalnie)

### Krok 1 — Przejdź do folderu implementacji i zainstaluj biblioteki

```bash
cd implementacja
pip install -r requirements.txt
```

### Krok 2 — Konfiguracja bazy danych

Uruchom lokalną instancję PostgreSQL i utwórz bazę danych:

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
python baza_danych/skrypty_migracji/import_trening.py
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

Aplikacja będzie dostępna lokalnie pod adresem: `http://localhost:8501`

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

Domyślny ciąg połączenia (przekazywany przez zmienną środowiskową `DB_URL` w `docker-compose.yml` lub domyślnie w kodzie):

```python
DB_URL = 'postgresql://postgres:postgres@db:5432/AuraMed3' # W Dockerze
DB_URL = 'postgresql://postgres:postgres@localhost:5432/AuraMed3' # Lokalnie
```

---

## Uwagi

- System ma charakter badawczo-edukacyjny i wspiera podejmowanie decyzji klinicznych, jednak ostateczna diagnoza zawsze należy do wykwalifikowanego personelu medycznego.