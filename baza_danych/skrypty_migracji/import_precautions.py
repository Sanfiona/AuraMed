import pandas as pd
from sqlalchemy import create_engine, text

# 1. Połączenie z bazą
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE choroba_zalecenia, zalecenia RESTART IDENTITY CASCADE;"))

diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)

# ==========================================
# CZĘŚĆ 1: Przetwarzanie precautions_df.csv
# ==========================================
df_prec = pd.read_csv('dane/precautions_df.csv')
all_prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']

# Pobieramy aktualny stan tabeli zaleceń z bazy
existing_advices = pd.read_sql('SELECT id, tresc_zalecenia FROM zalecenia', engine)

# Wyciągamy unikalne zalecenia z pliku precautions
all_unique_prec = pd.unique(df_prec[all_prec_cols].values.ravel())
all_unique_prec = [str(x).strip() for x in all_unique_prec if str(x) != 'nan']

# Filtrujemy tylko te, których jeszcze nie ma w bazie
really_new_prec = [a for a in all_unique_prec if a not in existing_advices['tresc_zalecenia'].values]

if really_new_prec:
    last_id = existing_advices['id'].max() if not existing_advices.empty else 0
    new_prec_df = pd.DataFrame({
        'id': range(last_id + 1, last_id + 1 + len(really_new_prec)),
        'tresc_zalecenia': really_new_prec
    })
    new_prec_df.to_sql('zalecenia', engine, if_exists='append', index=False)
    print(f"Dodano {len(really_new_prec)} zaleceń z precautions_df.csv.")

# Odświeżamy słownik z bazy po dodaniu nowych
full_advices_db = pd.read_sql('SELECT id, tresc_zalecenia FROM zalecenia', engine)

# Mapujemy i zapisujemy relacje dla precautions
melted_prec = df_prec.melt(id_vars=['Disease'], value_vars=all_prec_cols, value_name='Precaution')
melted_prec = melted_prec.dropna(subset=['Precaution'])
melted_prec['Precaution'] = melted_prec['Precaution'].str.strip()

mapping_prec = melted_prec.merge(diseases_db, left_on='Disease', right_on='nazwa_choroby')
mapping_prec = mapping_prec.merge(full_advices_db, left_on='Precaution', right_on='tresc_zalecenia')

final_prec_relation = mapping_prec[['id_x', 'id_y']].rename(columns={'id_x': 'choroba_id', 'id_y': 'zalecenie_id'})
final_prec_relation = final_prec_relation.drop_duplicates()
final_prec_relation.to_sql('choroba_zalecenia', engine, if_exists='append', index=False)


# ==========================================
# CZĘŚĆ 2: Przetwarzanie workout_df.csv
# ==========================================
df_work = pd.read_csv('../../dane/workout_df.csv')
print("Kolumny w workout_df.csv:", df_work.columns.tolist())
df_work['workout'] = df_work['workout'].str.strip()

# Odświeżamy słownik z bazy przed kolejnym sprawdzeniem
existing_advices = pd.read_sql('SELECT id, tresc_zalecenia FROM zalecenia', engine)

new_work_list = df_work['workout'].unique()
really_new_work = [a for a in new_work_list if a not in existing_advices['tresc_zalecenia'].values]

if really_new_work:
    last_id = existing_advices['id'].max()
    new_work_df = pd.DataFrame({
        'id': range(last_id + 1, last_id + 1 + len(really_new_work)),
        'tresc_zalecenia': really_new_work
    })
    new_work_df.to_sql('zalecenia', engine, if_exists='append', index=False)
    print(f"Dodano {len(really_new_work)} zaleceń z workout_df.csv.")

# Ponowne odświeżenie słownika z bazy po dodaniu workoutów
full_advices_db = pd.read_sql('SELECT id, tresc_zalecenia FROM zalecenia', engine)

# Mapujemy i zapisujemy relacje dla workout
mapping_work = df_work.merge(diseases_db, left_on='disease', right_on='nazwa_choroby')
mapping_work = mapping_work.merge(full_advices_db, left_on='workout', right_on='tresc_zalecenia')

final_work_relation = mapping_work[['id_x', 'id_y']].rename(columns={'id_x': 'choroba_id', 'id_y': 'zalecenie_id'})
final_work_relation = final_work_relation.drop_duplicates()
final_work_relation.to_sql('choroba_zalecenia', engine, if_exists='append', index=False)

print("Wszystkie zalecenia i relacje z obu plików zostały pomyślnie zmigrowane!")