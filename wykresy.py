import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Inicjalizacja połączenia z bazą danych PostgreSQL
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

# --- WYKRES 1: Rozkład diagnoz w danych historycznych ---
# USUNIĘTO LITERÓWKĘ: Zsynchronizowano alias kolumny na 'liczba_przypadkow'
query_dist = '''
    SELECT c.nazwa_choroby, COUNT(pt.id) as liczba_przypadkow
    FROM przypadki_treningowe pt
    JOIN choroby c ON pt.choroba_id = c.id
    GROUP BY c.nazwa_choroby
    ORDER BY liczba_przypadkow DESC
'''
df_dist = pd.read_sql(query_dist, engine)

plt.figure(figsize=(12, 8))
sns.barplot(data=df_dist, x='liczba_przypadkow', y='nazwa_choroby', hue='nazwa_choroby', palette='viridis', legend=False)
plt.title('Liczba przypadków historycznych dla każdej diagnozy')
plt.tight_layout()
plt.savefig('01_rozklad_diagnoz.png')
plt.show()

# --- WYKRES 2: Złożoność przypadków (Liczba objawów na pacjenta) ---
query_comp = '''
    SELECT przypadek_id, COUNT(objaw_id) as liczba_objawow
    FROM przypadek_objawy
    GROUP BY przypadek_id
'''
df_comp = pd.read_sql(query_comp, engine)

plt.figure(figsize=(10, 6))
sns.histplot(df_comp['liczba_objawow'], bins=range(1, 20), kde=True, color='purple')
plt.title('Średnia liczba objawów u pojedynczego pacjenta')
plt.xlabel('Liczba objawów')
plt.ylabel('Liczba pacjentów')
plt.savefig('02_zlozonosc_pacjentow.png')
plt.show()

# --- WYKRES 3: Heatmapa korelacji objawów ---
# USUNIĘTO LIMIT 1000: Zapytanie przetwarza teraz pełną bazę danych przypadków
query_matrix = '''
    SELECT po.przypadek_id, o.nazwa_objawu, 1 as wystepuje
    FROM przypadek_objawy po
    JOIN objawy o ON po.objaw_id = o.id
'''
df_matrix = pd.read_sql(query_matrix, engine)
pivot_df = df_matrix.pivot(index='przypadek_id', columns='nazwa_objawu', values='wystepuje').fillna(0)

# Korelacja dla 25 najczęstszych objawów
top_25 = df_matrix['nazwa_objawu'].value_counts().nlargest(25).index
corr_matrix = pivot_df[top_25].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.1)
plt.title('Korelacja współwystępowania 25 najczęstszych objawów')
plt.tight_layout()
plt.savefig('03_korelacja_objawow.png')
plt.show()

# --- WYKRES 4: Top 10 najczęstszych zaleceń ---
query_recs = '''
    SELECT z.tresc_zalecenia, COUNT(cz.choroba_id) as liczba_chorob
    FROM zalecenia z
    JOIN choroba_zalecenia cz ON z.id = cz.zalecenie_id
    GROUP BY z.tresc_zalecenia
    ORDER BY liczba_chorob DESC
    LIMIT 10
'''
df_recs = pd.read_sql(query_recs, engine)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_recs, x='liczba_chorob', y='tresc_zalecenia', palette='rocket', hue='tresc_zalecenia', legend=False)
plt.title('Najczęściej występujące zalecenia w systemie')
plt.savefig('04_top_zalecenia.png')
plt.show()

# --- WYKRES 5: Statystyki diet na chorobę ---
query_diets = '''
    SELECT c.nazwa_choroby, COUNT(cd.dieta_id) as n_diets
    FROM choroby c
    LEFT JOIN choroba_dieta cd ON c.id = cd.choroba_id
    GROUP BY c.nazwa_choroby
'''
df_diets = pd.read_sql(query_diets, engine)

plt.figure(figsize=(10, 5))
sns.boxplot(x=df_diets['n_diets'], color='cyan')
plt.title('Rozkład liczby przypisanych diet na jednostkę chorobową')
plt.xlabel('Liczba diet')
plt.savefig('05_box_diety.png')
plt.show()

# --- WYKRES 6: Macierz podobieństwa Jaccarda ---
query_jaccard = '''
    SELECT c.nazwa_choroby, o.nazwa_objawu
    FROM choroba_objawy co
    JOIN choroby c ON co.choroba_id = c.id
    JOIN objawy o ON co.objaw_id = o.id
'''
df_j = pd.read_sql(query_jaccard, engine)

sets = df_j.groupby('nazwa_choroby')['nazwa_objawu'].apply(set)
names = sets.index[:20]  # Ograniczenie do 20 chorób zachowane wyłącznie ze względu na czytelność etykiet na osiach
j_matrix = pd.DataFrame(0.0, index=names, columns=names)

for i in names:
    for j in names:
        u = len(sets[i].union(sets[j]))
        if u > 0:
            j_matrix.loc[i, j] = len(sets[i].intersection(sets[j])) / u

plt.figure(figsize=(12, 10))
sns.heatmap(j_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Podobieństwo profilu objawów między wybranymi chorobami')
plt.tight_layout()
plt.savefig('06_jaccard_similarity.png')
plt.show()

# --- WYKRES 7: Zależność między dotkliwością objawów a liczbą zaleceń ---
query = '''
    SELECT 
        c.id,
        AVG(o.waga) as srednia_waga_objawow,
        COUNT(DISTINCT cz.zalecenie_id) as liczba_zalecen,
        COUNT(DISTINCT cd.dieta_id) as liczba_diet
    FROM choroby c
    JOIN choroba_objawy co ON c.id = co.choroba_id
    JOIN objawy o ON co.objaw_id = o.id
    LEFT JOIN choroba_zalecenia cz ON c.id = cz.choroba_id
    LEFT JOIN choroba_dieta cd ON c.id = cd.choroba_id
    GROUP BY c.id
'''
df_corr = pd.read_sql(query, engine)

plt.figure(figsize=(10, 6))
sns.regplot(data=df_corr, x='srednia_waga_objawow', y='liczba_zalecen', color='teal')
plt.title('Zależność między dotkliwością objawów a liczbą zaleceń')
plt.xlabel('Średnia waga objawów choroby')
plt.ylabel('Liczba przypisanych zaleceń')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('07_korelacja_waga_zalecenia.png')
plt.show()