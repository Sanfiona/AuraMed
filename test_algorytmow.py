import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 1. Połączenie z bazą
# ─────────────────────────────────────────────
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

# ─────────────────────────────────────────────
# 2. Pobierz dane pacjentów
# ─────────────────────────────────────────────
query_p = '''
    SELECT po.przypadek_id, po.objaw_id, c.nazwa_choroby
    FROM przypadek_objawy po
    JOIN przypadki_treningowe pt ON po.przypadek_id = pt.id
    JOIN choroby c ON pt.choroba_id = c.id
'''
df_p = pd.read_sql(query_p, engine)

# ── POPRAWKA 1: prawidłowa macierz binarna ──
df_p['obecny'] = 1
X = df_p.pivot_table(
    index='przypadek_id',
    columns='objaw_id',
    values='obecny',
    fill_value=0
)
y = df_p.groupby('przypadek_id')['nazwa_choroby'].first()

# ── DIAGNOZA: sprawdź duplikaty ──
n_total = len(X)
n_unique = X.drop_duplicates().shape[0]
print(f"[INFO] Wszystkich przypadków : {n_total}")
print(f"[INFO] Unikalnych wierszy X  : {n_unique}")
print(f"[INFO] Duplikatów            : {n_total - n_unique}")
print(f"[INFO] Unikalnych chorób     : {y.nunique()}\n")

# ── POPRAWKA 2: usuń duplikaty (ten sam zestaw objawów + ta sama choroba) ──
df_combined = X.copy()
df_combined['_label'] = y
df_combined = df_combined.drop_duplicates()
y_clean = df_combined.pop('_label')
X_clean = df_combined

print(f"[INFO] Po usunięciu duplikatów: {len(X_clean)} przypadków\n")

# ── POPRAWKA 3: podział ze stratify + sprawdzenie przecięcia ──
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean,
    test_size=0.2,
    random_state=42,
    stratify=y_clean        # równy udział każdej choroby w obu zbiorach
)

# Weryfikacja: czy test set "wyciekł" z train set?
train_rows = set(map(tuple, X_train.values))
test_rows  = set(map(tuple, X_test.values))
overlap = len(train_rows & test_rows)
print(f"[WERYFIKACJA] Nakładające się wiersze train/test: {overlap}")
if overlap > 0:
    print("  ⚠️  UWAGA: wyciek danych! Wyniki będą zawyżone.")
else:
    print("  ✅ Brak wycieku danych.\n")

# ─────────────────────────────────────────────
# 3. Wiedza ekspercka (Cosine Similarity)
# ─────────────────────────────────────────────
query_e = '''
    SELECT c.nazwa_choroby, co.objaw_id, (co.czestotliwosc * o.waga) as sila
    FROM choroba_objawy co
    JOIN choroby c ON co.choroba_id = c.id
    JOIN objawy o ON co.objaw_id = o.id
'''
df_e = pd.read_sql(query_e, engine)
expert_matrix = (
    df_e
    .pivot_table(index='nazwa_choroby', columns='objaw_id', values='sila', fill_value=0)
    .reindex(columns=X_clean.columns, fill_value=0)
)

# ─────────────────────────────────────────────
# 4. Definicja modeli
# ─────────────────────────────────────────────
models = {
    "k-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes"         : GaussianNB(),
    "SGD Classifier"      : SGDClassifier(random_state=42, max_iter=1000),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# ─────────────────────────────────────────────
# 5. Ewaluacja: hold-out ORAZ cross-validation
# ─────────────────────────────────────────────
print("=" * 65)
print(f"{'Algorytm':<30} {'Hold-out':>9} {'CV mean':>9} {'CV std':>8}")
print("=" * 65)

for name, model in models.items():
    # Hold-out
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    holdout  = accuracy_score(y_test, y_pred)

    # Cross-validation (na całym zbiorze bez duplikatów)
    cv_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='accuracy')

    results.append({
        "Algorytm" : name,
        "Hold-out" : holdout,
        "CV mean"  : cv_scores.mean(),
        "CV std"   : cv_scores.std(),
        "Typ"      : "Machine Learning"
    })
    print(f"{name:<30} {holdout:>9.4f} {cv_scores.mean():>9.4f} {cv_scores.std():>8.4f}")

# ─────────────────────────────────────────────
# 6. Cosine Similarity (ekspercki)
# ─────────────────────────────────────────────
sims    = cosine_similarity(X_test, expert_matrix)
y_cos   = [expert_matrix.index[i] for i in sims.argmax(axis=1)]
cos_acc = accuracy_score(y_test, y_cos)

results.append({
    "Algorytm" : "Cosine Similarity (Wagi+Freq)",
    "Hold-out" : cos_acc,
    "CV mean"  : None,
    "CV std"   : None,
    "Typ"      : "Ekspercki"
})
print(f"{'Cosine Similarity (Wagi+Freq)':<30} {cos_acc:>9.4f} {'—':>9} {'—':>8}")
print("=" * 65)

# ─────────────────────────────────────────────
# 7. Tabela wynikowa
# ─────────────────────────────────────────────
df_results = (
    pd.DataFrame(results)
    .sort_values(by='Hold-out', ascending=False)
    .reset_index(drop=True)
)
print("\nTabela wynikowa:")
print(df_results.to_string(index=False))