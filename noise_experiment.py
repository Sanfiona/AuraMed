import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Połączenie z bazą i pobranie danych
# ─────────────────────────────────────────────
engine = create_engine('postgresql://postgres:postgres@localhost:5432/AuraMed3')

query_p = '''
    SELECT po.przypadek_id, po.objaw_id, c.nazwa_choroby
    FROM przypadek_objawy po
    JOIN przypadki_treningowe pt ON po.przypadek_id = pt.id
    JOIN choroby c ON pt.choroba_id = c.id
'''
df_p = pd.read_sql(query_p, engine)

df_p['obecny'] = 1
X = df_p.pivot_table(
    index='przypadek_id',
    columns='objaw_id',
    values='obecny',
    fill_value=0
)
y = df_p.groupby('przypadek_id')['nazwa_choroby'].first()

# Usuń duplikaty
df_combined = X.copy()
df_combined['_label'] = y
df_combined = df_combined.drop_duplicates()
y_clean = df_combined.pop('_label')
X_clean = df_combined

print(f"[INFO] Unikalnych przypadków: {len(X_clean)}")
print(f"[INFO] Unikalnych chorób: {y_clean.nunique()}\n")

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean,
    test_size=0.2,
    random_state=42,
    stratify=y_clean
)

# ─────────────────────────────────────────────
# 2. Funkcja wprowadzania szumu
# ─────────────────────────────────────────────
def wprowadz_szum(X, poziom_szumu, seed=42):
    """
    Wprowadza szum do danych binarnych przez losowe odwrócenie wartości.
    poziom_szumu: float np. 0.05 = 5%
    """
    np.random.seed(seed)
    X_noisy = X.copy().astype(float)
    n_total = X_noisy.size
    n_flip = int(n_total * poziom_szumu)
    
    # Losowo wybierz komórki do odwrócenia
    rows = np.random.randint(0, X_noisy.shape[0], n_flip)
    cols = np.random.randint(0, X_noisy.shape[1], n_flip)
    
    for r, c in zip(rows, cols):
        X_noisy.iloc[r, c] = 1 - X_noisy.iloc[r, c]
    
    return X_noisy

# ─────────────────────────────────────────────
# 3. Definicja modeli
# ─────────────────────────────────────────────
models = {
    "k-NN"               : KNeighborsClassifier(n_neighbors=5),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes"        : GaussianNB(),
    "SGD"                : SGDClassifier(random_state=42, max_iter=1000),
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

poziomy_szumu = [0, 0.05, 0.10, 0.20]
etykiety_szumu = ['0%', '5%', '10%', '20%']

# ─────────────────────────────────────────────
# 4. Główny eksperyment — testowanie z szumem
# ─────────────────────────────────────────────
print("=" * 70)
print("EKSPERYMENT: Wpływ szumu na dokładność algorytmów")
print("=" * 70)

# Słownik wyników: {nazwa_modelu: [acc_0%, acc_5%, acc_10%, acc_20%]}
wyniki_accuracy = {name: [] for name in models}
wyniki_f1       = {name: [] for name in models}
czasy_treningu  = {name: [] for name in models}

for poziom in poziomy_szumu:
    print(f"\n[SZUM: {int(poziom*100)}%]")
    
    if poziom == 0:
        X_train_noisy = X_train.copy()
    else:
        X_train_noisy = wprowadz_szum(X_train, poziom)
    
    for name, model in models.items():
        # Mierz czas treningu
        start = time.time()
        model.fit(X_train_noisy, y_train)
        czas = time.time() - start
        
        # Predykcja
        y_pred = model.predict(X_test)
        
        # Metryki
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        wyniki_accuracy[name].append(acc)
        wyniki_f1[name].append(f1)
        czasy_treningu[name].append(czas)
        
        print(f"  {name:<20} Accuracy: {acc:.4f}  F1: {f1:.4f}  Czas: {czas*1000:.1f}ms")

# ─────────────────────────────────────────────
# 5. Tabela wynikowa — metryki na czystych danych
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("METRYKI NA CZYSTYCH DANYCH (0% szumu)")
print("=" * 70)

metryki_czyste = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metryki_czyste.append({
        'Algorytm'  : name,
        'Accuracy'  : accuracy_score(y_test, y_pred),
        'Precision' : precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall'    : recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-score'  : f1_score(y_test, y_pred, average='weighted', zero_division=0),
    })

df_metryki = pd.DataFrame(metryki_czyste).set_index('Algorytm')
print(df_metryki.round(4).to_string())

# ─────────────────────────────────────────────
# 6. WYKRES 6.1 — Wpływ szumu na dokładność
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 7))

kolory = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
for (name, acc_list), kolor in zip(wyniki_accuracy.items(), kolory):
    plt.plot(etykiety_szumu, acc_list, marker='o', linewidth=2.5,
             markersize=8, label=name, color=kolor)

plt.title('Wpływ poziomu szumu na dokładność algorytmów klasyfikacyjnych',
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Poziom szumu', fontsize=12)
plt.ylabel('Dokładność (Accuracy)', fontsize=12)
plt.ylim(0, 1.05)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('wykres_6_1_szum_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("[ZAPISANO] wykres_6_1_szum_accuracy.png")

# ─────────────────────────────────────────────
# 7. WYKRES 6.2 — Heatmapa metryk na czystych danych
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.heatmap(
    df_metryki,
    annot=True,
    fmt='.3f',
    cmap='YlGnBu',
    linewidths=0.5,
    vmin=0.5,
    vmax=1.0
)
plt.title('Porównanie metryk algorytmów na czystych danych',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('wykres_6_2_heatmapa_metryk.png', dpi=150, bbox_inches='tight')
plt.show()
print("[ZAPISANO] wykres_6_2_heatmapa_metryk.png")

# ─────────────────────────────────────────────
# 8. WYKRES 6.3 — Feature Importance (Random Forest)
# ─────────────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Pobierz nazwy objawów z bazy
query_objawy = "SELECT id, nazwa_objawu FROM objawy ORDER BY id"
df_objawy = pd.read_sql(query_objawy, engine)
objaw_names = {row['id']: row['nazwa_objawu'] for _, row in df_objawy.iterrows()}

importances = pd.Series(
    rf_model.feature_importances_,
    index=[objaw_names.get(col, f'objaw_{col}') for col in X_train.columns]
).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 7))
sns.barplot(x=importances.values, y=importances.index, 
            palette='viridis', hue=importances.index, legend=False)
plt.title('Top 15 najważniejszych objawów diagnostycznych (Random Forest)',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Ważność cechy (Feature Importance)', fontsize=11)
plt.ylabel('Objaw', fontsize=11)
plt.tight_layout()
plt.savefig('wykres_6_3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("[ZAPISANO] wykres_6_3_feature_importance.png")

# ─────────────────────────────────────────────
# 9. WYKRES 6.4 — Czasy trenowania algorytmów
# ─────────────────────────────────────────────
czasy_czyste = {name: czasy[0] for name, czasy in czasy_treningu.items()}
df_czasy = pd.Series(czasy_czyste).sort_values(ascending=True) * 1000  # ms

plt.figure(figsize=(10, 5))
sns.barplot(x=df_czasy.values, y=df_czasy.index,
            palette='rocket', hue=df_czasy.index, legend=False)
plt.title('Czas trenowania algorytmów na czystych danych',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Czas trenowania [ms]', fontsize=11)
plt.ylabel('Algorytm', fontsize=11)
plt.tight_layout()
plt.savefig('wykres_6_4_czasy_trenowania.png', dpi=150, bbox_inches='tight')
plt.show()
print("[ZAPISANO] wykres_6_4_czasy_trenowania.png")

# ─────────────────────────────────────────────
# 10. WYKRES 6.5 — Spadek dokładności przy szumie (%)
# ─────────────────────────────────────────────
df_spadek = pd.DataFrame(wyniki_accuracy, index=etykiety_szumu).T
df_spadek['Spadek 20%'] = (df_spadek['0%'] - df_spadek['20%']) * 100

plt.figure(figsize=(10, 5))
df_spadek_sorted = df_spadek.sort_values('Spadek 20%', ascending=True)
sns.barplot(
    x=df_spadek_sorted['Spadek 20%'],
    y=df_spadek_sorted.index,
    palette='coolwarm_r',
    hue=df_spadek_sorted.index,
    legend=False
)
plt.title('Spadek dokładności przy 20% szumie względem danych czystych',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Spadek dokładności [punkty procentowe]', fontsize=11)
plt.ylabel('Algorytm', fontsize=11)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('wykres_6_5_spadek_dokladnosci.png', dpi=150, bbox_inches='tight')
plt.show()
print("[ZAPISANO] wykres_6_5_spadek_dokladnosci.png")

# ─────────────────────────────────────────────
# 11. Podsumowanie — tabela do wklejenia w pracę
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABELA WYNIKÓW — DO WKLEJENIA W PRACĘ")
print("=" * 70)
df_summary = pd.DataFrame(wyniki_accuracy, index=etykiety_szumu).T
df_summary.columns = ['Szum 0%', 'Szum 5%', 'Szum 10%', 'Szum 20%']
df_summary['Spadek'] = (df_summary['Szum 0%'] - df_summary['Szum 20%']).round(4)
print(df_summary.round(4).to_string())

print("\n[GOTOWE] Wszystkie wykresy zostały zapisane.")
