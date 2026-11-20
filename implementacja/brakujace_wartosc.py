from collections import Counter
import pandas as pd
import translations as t

df_train = pd.read_csv('dane/Training.csv')
df_meds = pd.read_csv('dane/medications.csv')

# 1. Sprawdzenie brakujących wartości w Training.csv
print('Brakujące wartości:', df_train.isnull().sum().sum())

# 2. Sprawdzenie, które 2 choroby z Training.csv nie pasują do medications.csv
choroby_train = set(df_train['prognosis'].str.strip().unique())
choroby_baza = set(df_meds['Disease'].unique())
print('Brak dopasowania dla:', choroby_train - choroby_baza)

# 3. Sprawdzenie duplikatów i wrażliwych haseł w translations.py
slownik = getattr(t, 'OBJAWY_PL', getattr(t, 'TRANSLATIONS', {}))
licznik = Counter(slownik.values())

print('\n=== DUPLIKATY TŁUMACZEŃ ===')
for pl, ile in licznik.items():
  if ile > 1:
    klucze = [k for k, v in slownik.items() if v == pl]
    print(f'{pl} -> {klucze}')

print('\n=== DO SPRAWDZENIA (NP. EXTRA_MARITAL_CONTACTS) ===')
for k, v in slownik.items():
  if any(
      w in k.lower() or w in v.lower()
      for w in ['marital', 'małż', 'sex', 'contact', 'kontakt']
  ):
    print(f'{k} -> {v}')