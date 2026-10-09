import pandas as pd
from sqlalchemy import create_engine, text

# 1. Połączenie
engine = create_engine('postgresql://postgres:koty@localhost:5432/AuraMed3')

# 2. Wczytanie pliku i czyszczenie nazw kolumn
df_train = pd.read_csv('dane/Training.csv')
# Czyścimy nazwy kolumn (małe litery, brak spacji), żeby pasowały do bazy
df_train.columns = [c.strip().replace(' ', '_').lower() for c in df_train.columns]

# 3. Pobranie słowników z bazy
diseases_db = pd.read_sql('SELECT id, nazwa_choroby FROM choroby', engine)
symptoms_db = pd.read_sql('SELECT id, nazwa_objawu FROM objawy', engine)

# Mapy do szybkiego wyszukiwania ID po nazwie
disease_map = dict(zip(diseases_db['nazwa_choroby'], diseases_db['id']))
symptom_map = dict(zip(symptoms_db['nazwa_objawu'], symptoms_db['id']))

# 4. Przetwarzanie danych
print("Rozpoczynam import... to może chwilę potrwać (4920 wierszy).")

with engine.begin() as conn:  # Używamy context managera dla transakcji
    for index, row in df_train.iterrows():
        # Kolumna z chorobą w Training.csv to zazwyczaj 'prognosis'
        disease_name = str(row['prognosis']).strip()
        d_id = disease_map.get(disease_name)
        
        if d_id:
            # A. Wstawiamy przypadek i pobieramy jego ID (używając text() dla bezpieczeństwa)
            query = text("INSERT INTO przypadki_treningowe (choroba_id) VALUES (:d_id) RETURNING id")
            res = conn.execute(query, {"d_id": d_id})
            przypadek_id = res.fetchone()[0]
            
            # B. Zbieramy objawy, które mają wartość 1
            active_symptoms = []
            for symptom_name, value in row.items():
                if value == 1:
                    # Szukamy ID objawu w naszym słowniku
                    s_id = symptom_map.get(symptom_name)
                    if s_id:
                        active_symptoms.append({
                            'przypadek_id': przypadek_id,
                            'objaw_id': s_id
                        })
            
            # C. Wstawiamy wszystkie objawy dla tego przypadku naraz
            if active_symptoms:
                sympt_df = pd.DataFrame(active_symptoms)
                sympt_df.to_sql('przypadek_objawy', conn, if_exists='append', index=False)
        
        if index % 500 == 0:
            print(f"Przetworzono {index} wierszy...")

print("Sukces! Dane treningowe zostały w pełni zmigrowane do bazy relacyjnej.")