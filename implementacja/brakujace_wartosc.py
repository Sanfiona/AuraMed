import pandas as pd

df_train = pd.read_csv('dane/Training.csv')
df_meds = pd.read_csv('dane/medications.csv')

# 1. Sprawdzenie brakujących wartości w Training.csv
print("Brakujące wartości:", df_train.isnull().sum().sum())

# 2. Sprawdzenie, które 2 choroby z Training.csv nie pasują do medications.csv
choroby_train = set(df_train['prognosis'].str.strip().unique())
choroby_baza = set(df_meds['Disease'].unique())
print("Brak dopasowania dla:", choroby_train - choroby_baza)