import pandas as pd
from sqlalchemy import create_engine

# 1. Połączenie
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

# 2. Wczytanie danych
# Plik zazwyczaj ma kolumny: Disease, Precaution_1, Precaution_2, Precaution_3, Precaution_4
df_prec = pd.read_csv('dane/precautions_df.csv')

# --- KROK A: Tabela zalecenia (Słownik unikalnych porad) ---
# Zbieramy wszystkie zalecenia ze wszystkich kolumn Precaution do jednej listy
all_prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
all_unique_prec = pd.unique(df_prec[all_prec_cols].values.ravel())
all_unique_prec = [x for x in all_unique_prec if str(x) != 'nan'] # Usuwamy puste wartości

prec_df = pd.DataFrame({'tresc_zalecenia': all_unique_prec}).reset_index().rename(columns={'index': 'id'})
prec_df['id'] += 1
prec_df['tresc_zalecenia'] = prec_df['tresc_zalecenia'].str.strip()

# Wysyłamy słownik zaleceń do bazy
prec_df.to_sql('zalecenia', engine, if_exists='replace', index=False)

# --- KROK B: Pobranie ID chorób ---
diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)

# --- KROK C: Tabela łącząca choroba_zalecenia ---
# Odwijamy tabelę (melt), żeby mieć pary: Choroba -> Zalecenie
melted_prec = df_prec.melt(id_vars=['Disease'], value_vars=all_prec_cols, value_name='Precaution')
melted_prec = melted_prec.dropna(subset=['Precaution'])
melted_prec['Precaution'] = melted_prec['Precaution'].str.strip()

# Łączymy po nazwie z ID chorób
mapping_prec = melted_prec.merge(diseases_db, left_on='Disease', right_on='nazwa_choroby')

# Łączymy po treści z ID zaleceń
mapping_prec = mapping_prec.merge(prec_df, left_on='Precaution', right_on='tresc_zalecenia')

# Wybieramy ID (id_x - choroba, id_y - zalecenie)
final_prec_relation = mapping_prec[['id_x', 'id_y']].rename(columns={'id_x': 'choroba_id', 'id_y': 'zalecenie_id'})

# Usuwamy ewentualne duplikaty
final_prec_relation = final_prec_relation.drop_duplicates()

# Wysyłamy do Postgresa
final_prec_relation.to_sql('choroba_zalecenia', engine, if_exists='replace', index=False)

print("Zalecenia zostały pomyślnie zmigrowane!")