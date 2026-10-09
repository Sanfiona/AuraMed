import pandas as pd
from sqlalchemy import create_engine

# 1. Konfiguracja połączenia z Postgres
engine = create_engine('postgresql://postgres:koty@localhost:5432/AuraMed3')

# 2. Wczytanie danych
df_symptoms = pd.read_csv('dane/symtoms_df.csv')
df_severity = pd.read_csv('dane/Symptom-severity.csv')


# --- KROK 1: TABELA OBJAWY (Tworzymy ją na nowo z wagami) ---
df_severity['Symptom'] = df_severity['Symptom'].str.strip().str.replace(' ', '_')
objawy_df = df_severity.rename(columns={'Symptom': 'nazwa_objawu', 'weight': 'waga'})

# Sprawdzenie i wypisanie duplikatów przed zapisem
duplikaty = objawy_df[objawy_df.duplicated(subset=['nazwa_objawu'], keep=False)]
print("Znalezione duplikaty w pliku:")
print(duplikaty[['nazwa_objawu']])

# Usunięcie duplikatów zapobiegające błędom unikalności w bazie
objawy_df = objawy_df.drop_duplicates(subset=['nazwa_objawu']).reset_index(drop=True)
objawy_df['id'] = objawy_df.index + 1

# Wysyłamy słownik objawów do bazy
objawy_df.to_sql('objawy', engine, if_exists='append', index=False)

# --- KROK 2: POBIERANIE ISTNIEJĄCYCH CHORÓB ---
diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)

# --- KROK 3: OBLICZANIE CZĘSTOTLIWOŚCI ---
disease_counts = df_symptoms['Disease'].value_counts().to_dict()

melted_df = df_symptoms.melt(id_vars=['Disease'], value_name='Symptom').dropna(subset=['Symptom'])
melted_df['Symptom'] = melted_df['Symptom'].str.strip().str.replace(' ', '_')

freq_df = melted_df.groupby(['Disease', 'Symptom']).size().reset_index(name='count')

freq_df['czestotliwosc'] = freq_df.apply(
    lambda row: round(row['count'] / disease_counts[row['Disease']], 2) 
    if row['Disease'] in disease_counts else 0, axis=1
)

# --- KROK 4: TWORZENIE TABELI ŁĄCZĄCEJ ---
mapping = freq_df.merge(diseases_db, left_on='Disease', right_on='nazwa_choroby')
mapping = mapping.merge(objawy_df, left_on='Symptom', right_on='nazwa_objawu')

final_relation = mapping[['id_x', 'id_y', 'czestotliwosc']].rename(
    columns={'id_x': 'choroba_id', 'id_y': 'objaw_id'}
)

final_relation.to_sql('choroba_objawy', engine, if_exists='append', index=False)

print("Gotowe! Tabela 'choroby' została nienaruszona, a relacje i objawy zostały dopisane.")