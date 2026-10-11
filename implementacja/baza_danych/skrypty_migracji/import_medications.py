import os

import pandas as pd
from sqlalchemy import create_engine, text


# 1. Konfiguracja połączenia z Postgres
db_url = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/AuraMed3")
engine = create_engine(db_url)

with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE choroby, leki, choroba_lek RESTART IDENTITY CASCADE;"))
# 2. Wczytanie danych
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'dane', 'medications.csv'))
df_meds = pd.read_csv(csv_path)
csv_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'dane', 'description.csv'))
df_desc = pd.read_csv(csv_path) 

# Czyszczenie leków (tak jak wcześniej)
df_meds['Medication'] = df_meds['Medication'].str.replace(r"[\[\]']", "", regex=True).str.split(", ")

# --- KROK A: Tabela leków (Słownik) ---
all_meds = df_meds['Medication'].explode().unique()
meds_df = pd.DataFrame({'nazwa_leku': all_meds}).reset_index().rename(columns={'index': 'id'})
meds_df['id'] += 1
meds_df.to_sql('leki', engine, if_exists='append', index=False)

# --- KROK B: Tabela chorób Z OPISAMI ---
# 1. Wyciągamy unikalne nazwy chorób
unique_diseases = pd.DataFrame({'nazwa_choroby': df_meds['Disease'].unique()})

# 2. Łączymy nazwy chorób z ich opisami z drugiego pliku
# left_on='nazwa_choroby', right_on='Disease' (dostosuj nazwy kolumn jeśli są inne)
diseases_with_desc = unique_diseases.merge(df_desc, left_on='nazwa_choroby', right_on='Disease', how='left')

# 3. Czyścimy i dodajemy ID
diseases_df = diseases_with_desc[['nazwa_choroby', 'Description']].rename(columns={'Description': 'opis'})
diseases_df = diseases_df.reset_index().rename(columns={'index': 'id'})
diseases_df['id'] += 1

# Wysyłamy tabelę z kolumną 'opis' do Postgresa
diseases_df.to_sql('choroby', engine, if_exists='append', index=False)

# --- KROK C: Tabela łącząca choroba_lek ---
exploded_df = df_meds.explode('Medication')

# Łączymy z ID chorób i ID leków (używając nowej tabeli diseases_df)
mapping_df = exploded_df.merge(diseases_df, left_on='Disease', right_on='nazwa_choroby')
mapping_df = mapping_df.merge(meds_df, left_on='Medication', right_on='nazwa_leku')

# Wybieramy ID (id_x to ID choroby, id_y to ID leku)
final_relation_df = mapping_df[['id_x', 'id_y']].rename(columns={'id_x': 'choroba_id', 'id_y': 'lek_id'})

final_relation_df.to_sql('choroba_lek', engine, if_exists='append', index=False)

print("Migracja z opisami zakończona sukcesem!")