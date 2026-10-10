import os

import pandas as pd
from sqlalchemy import create_engine, text


# 1. Konfiguracja połączenia
db_url = os.getenv("DB_URL", "postgresql://postgres:koty@localhost:5432/AuraMed3")
engine = create_engine(db_url)
# Wyczyszczenie tabel i zresetowanie liczników ID przed ponownym wstawieniem
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE choroba_dieta, diety RESTART IDENTITY CASCADE;"))

# 2. Wczytanie danych
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'dane', 'diets.csv'))
df_diets = pd.read_csv(csv_path)

# Czyszczenie i rozbijanie listy tekstowej na prawdziwą listę
df_diets['Diet'] = df_diets['Diet'].str.replace(r"[\[\]']", "", regex=True).str.split(",")

# --- KROK A: Tabela diety (Słownik) ---
all_diets = df_diets['Diet'].explode().str.strip().unique()
diets_df = pd.DataFrame({'nazwa_diety': all_diets}).reset_index().rename(columns={'index': 'id'})
diets_df['id'] += 1
diets_df.to_sql('diety', engine, if_exists='append', index=False)

# --- KROK B: Pobranie chorób ---
diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)

# --- KROK C: Tabela łącząca choroba_dieta ---
exploded_df = df_diets.explode('Diet')
exploded_df['Diet'] = exploded_df['Diet'].str.strip()

mapping_df = exploded_df.merge(diseases_db, left_on='Disease', right_on='nazwa_choroby')
mapping_df = mapping_df.merge(diets_df, left_on='Diet', right_on='nazwa_diety')

final_relation_df = mapping_df[['id_x', 'id_y']].rename(columns={'id_x': 'choroba_id', 'id_y': 'dieta_id'})
final_relation_df = final_relation_df.drop_duplicates()

final_relation_df.to_sql('choroba_dieta', engine, if_exists='append', index=False)

print("Migracja diet zakończona sukcesem!")