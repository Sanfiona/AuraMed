"""
train_model.py
Uruchom ten skrypt PRZED uruchomieniem aplikacji Streamlit.
Trenuje model przy użyciu wag oraz częstotliwości i zapisuje wszystkie potrzebne pliki .pkl
"""
import os

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────
# 1. Połączenie z bazą i pobranie danych
# ─────────────────────────────────────────────
db_url = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/AuraMed3")
engine = create_engine(db_url)

print("[INFO] Pobieranie przypadków treningowych z bazy...")
query_przypadki = '''
    SELECT po.przypadek_id, po.objaw_id, c.nazwa_choroby, pt.choroba_id
    FROM przypadek_objawy po
    JOIN przypadki_treningowe pt ON po.przypadek_id = pt.id
    JOIN choroby c ON pt.choroba_id = c.id
'''
df = pd.read_sql(query_przypadki, engine)

print("[INFO] Pobieranie wag objawów...")
df_wagi = pd.read_sql("SELECT id, waga FROM objawy", engine)
wagi_dict = dict(zip(df_wagi['id'], df_wagi['waga']))

print("[INFO] Pobieranie częstotliwości występowania objawów...")
# UWAGA: Jeśli Twoja tabela łącząca chorobę z objawem (ta z kolumną czestotliwosc) 
# nazywa się inaczej niż 'choroba_objawy', zmień jej nazwę poniżej:
query_czest = "SELECT choroba_id, objaw_id, czestotliwosc FROM choroba_objawy"
df_czestotliwosci = pd.read_sql(query_czest, engine)

# Tworzymy słownik mapujący parę (choroba_id, objaw_id) -> czestotliwosc
czest_dict = dict(zip(
    zip(df_czestotliwosci['choroba_id'], df_czestotliwosci['objaw_id']), 
    df_czestotliwosci['czestotliwosc']
))

def oblicz_score_objawu(row):
    return float(wagi_dict.get(row['objaw_id'],1))

df['wartość_cechy'] = df.apply(oblicz_score_objawu, axis=1)

# Tworzymy tabelę przestawną (pivot) na podstawie wyliczonych wag i częstotliwości
X = df.pivot_table(
    index='przypadek_id',
    columns='objaw_id',
    values='wartość_cechy',
    fill_value=0.0
)
y = df.groupby('przypadek_id')['nazwa_choroby'].first()

# Usuwanie duplikatów ze zbioru
df_combined = X.copy()
df_combined['_label'] = y
df_combined = df_combined.drop_duplicates()
y_clean = df_combined.pop('_label')
X_clean = df_combined

print(f"[INFO] Przypadków treningowych po oczyszczeniu: {len(X_clean)}")
print(f"[INFO] Liczba objawów (cech wejściowych): {X_clean.shape[1]}")
print(f"[INFO] Liczba unikalnych chorób: {y_clean.nunique()}")

# ─────────────────────────────────────────────
# 2. Trenowanie modeli uczenia maszynowego
# ─────────────────────────────────────────────
print("\n[INFO] Rozpoczynanie treningu modeli...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_clean, y_clean)
print("[INFO] Model Logistic Regression został wytrenowany.")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_clean, y_clean)
print("[INFO] Model Random Forest został wytrenowany.")

# ─────────────────────────────────────────────
# 3. Przygotowanie nazw objawów do aplikacji
# ─────────────────────────────────────────────
query_objawy_nazwy = "SELECT id, nazwa_objawu FROM objawy ORDER BY id"
df_objawy_nazwy = pd.read_sql(query_objawy_nazwy, engine)
objaw_id_to_name = dict(zip(df_objawy_nazwy['id'], df_objawy_nazwy['nazwa_objawu']))

# Mapowanie ID kolumn z X_clean na ich czytelne angielskie nazwy
feature_names = [objaw_id_to_name.get(col, f'objaw_{col}') for col in X_clean.columns]

# ─────────────────────────────────────────────
# 4. Eksport i zapis plików .pkl
# ─────────────────────────────────────────────
print("\n[INFO] Zapisywanie modeli i plików konfiguracyjnych...")
joblib.dump(lr_model, 'model_lr.pkl')
joblib.dump(rf_model, 'model_rf.pkl')
joblib.dump(list(X_clean.columns), 'feature_ids.pkl')     # Lista ID kolumn (potrzebna do wektora wejściowego)
joblib.dump(feature_names, 'feature_names.pkl')           # Polskie/angielskie nazwy mapowane w Streamlit
joblib.dump(list(y_clean.unique()), 'classes.pkl')        # Lista wszystkich obsługiwanych chorób

print("\n[GOTOWE] Skrypt zakończył działanie sukcesem! Zapisano pliki:")
print("  - model_lr.pkl")
print("  - model_rf.pkl")
print("  - feature_ids.pkl")
print("  - feature_names.pkl")
print("  - classes.pkl")
print("[WSKAZÓWKA] Możesz teraz bezpiecznie uruchomić aplikację: streamlit run app.py")