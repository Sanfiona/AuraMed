import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

# 1. Wczytujemy ten drugi plik (z poradami o jedzeniu/ćwiczeniach)
df_work = pd.read_csv('dane/workout_df.csv')
# Zakładam że kolumna z poradą nazywa się 'workout'
df_work['workout'] = df_work['workout'].str.strip()

# 2. Pobieramy aktualny słownik zaleceń z bazy
existing_advices = pd.read_sql('SELECT id, tresc_zalecenia FROM zalecenia', engine)

# --- AKTUALIZACJA SŁOWNIKA ---
# Sprawdzamy, których porad z nowego pliku nie ma jeszcze w bazie
new_advices_list = df_work['workout'].unique()
really_new = [a for a in new_advices_list if a not in existing_advices['tresc_zalecenia'].values]

if really_new:
    last_id = existing_advices['id'].max()
    new_advices_df = pd.DataFrame({
        'id': range(last_id + 1, last_id + 1 + len(really_new)),
        'tresc_zalecenia': really_new
    })
    # Dopisujemy NOWE treści do słownika
    new_advices_df.to_sql('zalecenia', engine, if_exists='append', index=False)
    print(f"Dodano {len(really_new)} nowych unikalnych porad do słownika.")

# --- AKTUALIZACJA RELACJI (Tabela łącząca) ---
# Pobieramy odświeżony słownik i ID chorób
full_advices_db = pd.read_sql('SELECT id, tresc_zalecenia FROM zalecenia', engine)
diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)

# Mapujemy nowy plik na ID
mapping = df_work.merge(diseases_db, left_on='disease', right_on='nazwa_choroby')
mapping = mapping.merge(full_advices_db, left_on='workout', right_on='tresc_zalecenia')

final_new_relations = mapping[['id_x', 'id_y']].rename(columns={'id_x': 'choroba_id', 'id_y': 'zalecenie_id'})

# Dopisujemy nowe relacje do tabeli łączącej (if_exists='append')
# UWAGA: 'append' zadziała jeśli nie masz nałożonego klucza UNIQUE na pary
final_new_relations.to_sql('choroba_zalecenia', engine, if_exists='append', index=False)

print("Nowe zalecenia zostały dopisane do istniejącej tabeli!")