import pandas as pd
from sqlalchemy import create_engine

# 1. Konfiguracja połączenia z Postgres
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

# 2. Wczytanie danych
df_symptoms = pd.read_csv('dane/symtoms_df.csv')
df_severity = pd.read_csv('dane/Symptom-severity.csv') # Zakładamy, że ten plik ma kolumny 'Disease' i 'Description'


# --- KROK 1: TABELA OBJAWY (Tworzymy ją na nowo z wagami) ---
df_severity['Symptom'] = df_severity['Symptom'].str.strip().str.replace(' ', '_')
objawy_df = df_severity.rename(columns={'Symptom': 'nazwa_objawu', 'weight': 'waga'})
objawy_df = objawy_df.reset_index().rename(columns={'index': 'id'})
objawy_df['id'] += 1

# Wysyłamy słownik objawów do bazy
objawy_df.to_sql('objawy', engine, if_exists='replace', index=False)

# --- KROK 2: POBIERANIE ISTNIEJĄCYCH CHORÓB ---
# Pobieramy to, co już masz w Postgresie, żeby znać ich ID
diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)

# --- KROK 3: OBLICZANIE CZĘSTOTLIWOŚCI (Z pacjentów w Excelu) ---
# Liczymy ile razy dana choroba występuje w pliku
disease_counts = df_symptoms['Disease'].value_counts().to_dict()

# "Odwijamy" pacjentów na format pionowy
melted_df = df_symptoms.melt(id_vars=['Disease'], value_name='Symptom').dropna(subset=['Symptom'])
melted_df['Symptom'] = melted_df['Symptom'].str.strip().str.replace(' ', '_')

# Liczymy wystąpienia par Choroba-Objaw
freq_df = melted_df.groupby(['Disease', 'Symptom']).size().reset_index(name='count')

# Obliczamy czestotliwosc (liczba wystąpień objawu / liczba wystąpień choroby)
freq_df['czestotliwosc'] = freq_df.apply(
    lambda row: round(row['count'] / disease_counts[row['Disease']], 2) 
    if row['Disease'] in disease_counts else 0, axis=1
)

# --- KROK 4: TWORZENIE TABELI ŁĄCZĄCEJ ---
# Łączymy dane z ID chorób z bazy
mapping = freq_df.merge(diseases_db, left_on='Disease', right_on='nazwa_choroby')
# Łączymy dane z ID objawów które przed chwilą stworzyliśmy
mapping = mapping.merge(objawy_df, left_on='Symptom', right_on='nazwa_objawu')

# Wybieramy tylko kolumny do tabeli łączącej
# id_x to ID z Twojej istniejącej tabeli choroby, id_y to ID z nowej tabeli objawy
final_relation = mapping[['id_x', 'id_y', 'czestotliwosc']].rename(
    columns={'id_x': 'choroba_id', 'id_y': 'objaw_id'}
)

# Wysyłamy do Postgresa
final_relation.to_sql('choroba_objawy', engine, if_exists='replace', index=False)

print("Gotowe! Tabela 'choroby' została nienaruszona, a relacje i objawy zostały dopisane.")