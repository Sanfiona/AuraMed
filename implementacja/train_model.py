"""
train_model.py
Uruchom ten skrypt PRZED uruchomieniem aplikacji Streamlit.
Trenuje model i zapisuje wszystkie potrzebne pliki .pkl
"""
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. Połączenie z bazą i pobranie danych
# ─────────────────────────────────────────────
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

query = '''
    SELECT po.przypadek_id, po.objaw_id, c.nazwa_choroby
    FROM przypadek_objawy po
    JOIN przypadki_treningowe pt ON po.przypadek_id = pt.id
    JOIN choroby c ON pt.choroba_id = c.id
'''
df = pd.read_sql(query, engine)

df['obecny'] = 1
X = df.pivot_table(
    index='przypadek_id',
    columns='objaw_id',
    values='obecny',
    fill_value=0
)
y = df.groupby('przypadek_id')['nazwa_choroby'].first()

# Usuń duplikaty
df_combined = X.copy()
df_combined['_label'] = y
df_combined = df_combined.drop_duplicates()
y_clean = df_combined.pop('_label')
X_clean = df_combined

print(f"[INFO] Przypadków treningowych: {len(X_clean)}")
print(f"[INFO] Liczba objawów (cech): {X_clean.shape[1]}")
print(f"[INFO] Liczba chorób: {y_clean.nunique()}")

# ─────────────────────────────────────────────
# 2. Trenowanie modeli
# ─────────────────────────────────────────────
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_clean, y_clean)
print("[INFO] Logistic Regression wytrenowana")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_clean, y_clean)
print("[INFO] Random Forest wytrenowany")

# ─────────────────────────────────────────────
# 3. Pobierz nazwy objawów z bazy
# ─────────────────────────────────────────────
query_objawy = "SELECT id, nazwa_objawu, waga FROM objawy ORDER BY id"
df_objawy = pd.read_sql(query_objawy, engine)
objaw_id_to_name = dict(zip(df_objawy['id'], df_objawy['nazwa_objawu']))

# Mapa: nazwa_objawu -> kolumna w X_clean
feature_names = [objaw_id_to_name.get(col, f'objaw_{col}') for col in X_clean.columns]

# ─────────────────────────────────────────────
# 4. Zapisz wszystkie pliki
# ─────────────────────────────────────────────
joblib.dump(lr_model, 'model_lr.pkl')
joblib.dump(rf_model, 'model_rf.pkl')
joblib.dump(list(X_clean.columns), 'feature_ids.pkl')     # ID kolumn
joblib.dump(feature_names, 'feature_names.pkl')           # Nazwy objawów
joblib.dump(list(y_clean.unique()), 'classes.pkl')        # Lista chorób

print("\n[GOTOWE] Zapisano pliki:")
print("  - model_lr.pkl")
print("  - model_rf.pkl")
print("  - feature_ids.pkl")
print("  - feature_names.pkl")
print("  - classes.pkl")
