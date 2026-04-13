import pandas as pd
from sqlalchemy import create_engine

# 1. Konfiguracja połączenia
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

# 2. Wczytanie danych
df_diets = pd.read_csv('dane/diets.csv')

# Czyszczenie i rozbijanie listy tekstowej na prawdziwą listę
df_diets['Diet'] = df_diets['Diet'].str.replace(r"[\[\]']", "", regex=True).str.split(",")

# --- KROK A: Tabela diety (Słownik) ---
# Explode, strip (usunięcie spacji) i unique
all_diets = df_diets['Diet'].explode().str.strip().unique()
diets_df = pd.DataFrame({'nazwa_diety': all_diets}).reset_index().rename(columns={'index': 'id'})
diets_df['id'] += 1
diets_df.to_sql('diety', engine, if_exists='replace', index=False)

# --- KROK B: Pobranie chorób ---
diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)

# --- KROK C: Tabela łącząca choroba_dieta ---
exploded_df = df_diets.explode('Diet')
exploded_df['Diet'] = exploded_df['Diet'].str.strip() # Ważne dla czystości merge'a

# Łączymy z ID chorób (dostaniemy kolumnę 'id' z diseases_db)
mapping_df = exploded_df.merge(diseases_db, left_on='Disease', right_on='nazwa_choroby')

# Łączymy z ID diet (id z poprzedniego kroku stanie się id_x, id z diets_df stanie się id_y)
mapping_df = mapping_df.merge(diets_df, left_on='Diet', right_on='nazwa_diety')

# Wybieramy ID i zmieniamy nazwy na docelowe
final_relation_df = mapping_df[['id_x', 'id_y']].rename(columns={'id_x': 'choroba_id', 'id_y': 'dieta_id'})

# Usuwamy duplikaty, jeśli dieta powtarzała się w wierszach dla tej samej choroby
final_relation_df = final_relation_df.drop_duplicates()

final_relation_df.to_sql('choroba_dieta', engine, if_exists='replace', index=False)

print("Migracja diet zakończona sukcesem!")