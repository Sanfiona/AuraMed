import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
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

# Podział
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean,
    test_size=0.2,
    random_state=42,
    stratify=y_clean
)

# ─────────────────────────────────────────────
# 2. Binaryzacja etykiet (wymagane dla ROC multiclass)
# ─────────────────────────────────────────────
classes = sorted(y_clean.unique())
y_test_bin = label_binarize(y_test, classes=classes)

# ─────────────────────────────────────────────
# 3. Definicja modeli z predict_proba
# ─────────────────────────────────────────────
models = {
    "k-NN"               : OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
    "Random Forest"      : OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Naive Bayes"        : OneVsRestClassifier(GaussianNB()),
    "SGD"                : OneVsRestClassifier(SGDClassifier(random_state=42, max_iter=1000, loss='modified_huber')),
    "Decision Tree"      : OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
}

kolory = {
    "k-NN"               : '#2196F3',
    "Random Forest"      : '#4CAF50',
    "Naive Bayes"        : '#FF9800',
    "SGD"                : '#E91E63',
    "Decision Tree"      : '#9C27B0',
    "Logistic Regression": '#00BCD4',
}

# ─────────────────────────────────────────────
# 4. Obliczanie AUC i rysowanie krzywych ROC
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 8))

print("=" * 50)
print(f"{'Algorytm':<25} {'AUC (macro)'}")
print("=" * 50)

auc_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)

    # Macro-average AUC
    auc_score = roc_auc_score(
        y_test_bin, y_prob,
        multi_class='ovr',
        average='macro'
    )
    auc_results[name] = auc_score
    print(f"{name:<25} {auc_score:.4f}")

    # Micro-average ROC curve (do wykresu)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    plt.plot(fpr, tpr,
             label=f"{name} (AUC = {auc_score:.3f})",
             color=kolory[name],
             linewidth=2)

print("=" * 50)

# Linia losowego klasyfikatora
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Losowy klasyfikator (AUC = 0.500)')

plt.title('Krzywa ROC dla algorytmów klasyfikacyjnych\n(podejście One-vs-Rest, micro-average)',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Odsetek fałszywie pozytywnych (False Positive Rate)', fontsize=11)
plt.ylabel('Odsetek prawdziwie pozytywnych (True Positive Rate)', fontsize=11)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('wykres_6_3_roc.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[ZAPISANO] wykres_6_3_roc.png")

# ─────────────────────────────────────────────
# 5. Tabela AUC do wklejenia w pracę
# ─────────────────────────────────────────────
print("\nTabela AUC — do wklejenia w pracę:")
df_auc = pd.DataFrame.from_dict(auc_results, orient='index', columns=['AUC (macro)'])
df_auc = df_auc.sort_values('AUC (macro)', ascending=False)
print(df_auc.round(4).to_string())
