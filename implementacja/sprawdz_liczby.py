import numpy as np, pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')
q = '''
SELECT po.przypadek_id, po.objaw_id, c.nazwa_choroby
FROM przypadek_objawy po
JOIN przypadki_treningowe pt ON po.przypadek_id = pt.id
JOIN choroby c ON pt.choroba_id = c.id
'''
df = pd.read_sql(q, engine)
df['obecny'] = 1
X = df.pivot_table(index='przypadek_id', columns='objaw_id',
                   values='obecny', fill_value=0)
y = df.groupby('przypadek_id')['nazwa_choroby'].first()

print("Wszystkie przypadki:", len(X), "| kolumn (objawów):", X.shape[1])
print("Średnia liczba objawów (wszystkie):", round(X.sum(axis=1).mean(), 2))

d = X.copy(); d['_y'] = y
Xu = d.drop_duplicates().drop(columns='_y')
n = Xu.sum(axis=1)
print("Unikalne:", len(Xu), "| średnia:", round(n.mean(), 2),
      "| mediana:", n.median(), "| min:", n.min(), "| max:", n.max())

def wprowadz_szum(X, poziom, seed=42):      # kopia z noise_experiment.py
    np.random.seed(seed)
    Xn = X.copy().astype(float)
    k = int(Xn.size * poziom)
    rows = np.random.randint(0, Xn.shape[0], k)
    cols = np.random.randint(0, Xn.shape[1], k)
    for r, c in zip(rows, cols):
        Xn.iloc[r, c] = 1 - Xn.iloc[r, c]
    return Xn

for p in [0.05, 0.10, 0.20]:
    diff = wprowadz_szum(Xu, p).values - Xu.values
    print(f"Szum {int(p*100)}%: dodane objawy (0→1) na wiersz =",
          round((diff == 1).sum(axis=1).mean(), 2),
          "| usunięte (1→0) =", round((diff == -1).sum(axis=1).mean(), 2))