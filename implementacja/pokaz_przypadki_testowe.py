"""
pokaz_przypadki_testowe.py
Wypisuje przykładowe przypadki testowe dla wybranych chorób z Training.csv
"""
import pandas as pd

# Wczytaj plik testowy
df = pd.read_csv('../dane/Training.csv')
print(df['prognosis'].unique())

# Wybrane choroby do testowania
choroby = ['Diabetes ', 'Heart attack', 'Migraine']

for choroba in choroby:
    print(f"\n{'='*60}")
    print(f"CHOROBA: {choroba}")
    print(f"{'='*60}")
    
    # Znajdź przypadki tej choroby
    przypadki = df[df['prognosis'] == choroba]
    
    if przypadki.empty:
        print(f"Brak przypadków dla: {choroba}")
        continue
    
    # Weź pierwszy przypadek
    przypadek = przypadki.iloc[0]
    
    # Wypisz tylko obecne objawy (wartość = 1)
    objawy_obecne = [col for col in df.columns 
                     if col != 'prognosis' and przypadek[col] == 1]
    
    print(f"Liczba przypadków w Testing.csv: {len(przypadki)}")
    print(f"\nObjawy (do wpisania w aplikacji):")
    for i, objaw in enumerate(objawy_obecne, 1):
        print(f"  {i}. {objaw.replace('_', ' ')}")
    
    print(f"\nOczekiwana diagnoza: {choroba}")
    print(f"Liczba objawów: {len(objawy_obecne)}")
