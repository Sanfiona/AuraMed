"""
app.py - Aplikacja AuraMed
Uruchom: streamlit run app.py
Wymagane: najpierw uruchom train_model.py
"""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from translations import tlumacz_objaw, tlumacz_diete, tlumacz_chorobe, tlumacz_zalecenie, przygotuj_liste_objawow
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AuraMed — System Rekomendacji Terapii",
    page_icon="🏥",
    layout="wide"
)

DB_URL = 'postgresql://postgres:postgres@localhost:5432/AuraMed3'

# ─────────────────────────────────────────────
# ŁADOWANIE ZASOBÓW (cache)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        'lr': joblib.load('model_lr.pkl'),
        'rf': joblib.load('model_rf.pkl'),
        'feature_ids': joblib.load('feature_ids.pkl'),
        'feature_names': joblib.load('feature_names.pkl'),
    }

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

def load_wagi():
    engine = get_engine()
    df_wagi = pd.read_sql("SELECT id, waga FROM objawy", engine)
    return dict(zip(df_wagi['id'], df_wagi['waga']))

@st.cache_data
def load_training_data():
    engine = get_engine()
    query = '''
        SELECT po.przypadek_id, po.objaw_id, c.nazwa_choroby
        FROM przypadek_objawy po
        JOIN przypadki_treningowe pt ON po.przypadek_id = pt.id
        JOIN choroby c ON pt.choroba_id = c.id
    '''
    df = pd.read_sql(query, engine)
    df['obecny'] = 1
    X = df.pivot_table(
        index='przypadek_id',
        columns='objaw_id',
        values='obecny',
        fill_value=0
    )
    y = df.groupby('przypadek_id')['nazwa_choroby'].first()
    df_combined = X.copy()
    df_combined['_label'] = y
    df_combined = df_combined.drop_duplicates()
    y_clean = df_combined.pop('_label')
    return df_combined, y_clean

@st.cache_data
def get_recommendations(disease_name):
    engine = get_engine()
    leki = pd.read_sql(f"""
        SELECT l.nazwa_leku FROM leki l
        JOIN choroba_lek cl ON l.id = cl.lek_id
        JOIN choroby c ON cl.choroba_id = c.id
        WHERE c.nazwa_choroby = '{disease_name}'
    """, engine)

    diety = pd.read_sql(f"""
        SELECT d.nazwa_diety FROM diety d
        JOIN choroba_dieta cd ON d.id = cd.dieta_id
        JOIN choroby c ON cd.choroba_id = c.id
        WHERE c.nazwa_choroby = '{disease_name}'
    """, engine)

    zalecenia = pd.read_sql(f"""
        SELECT z.tresc_zalecenia FROM zalecenia z
        JOIN choroba_zalecenia cz ON z.id = cz.zalecenie_id
        JOIN choroby c ON cz.choroba_id = c.id
        WHERE c.nazwa_choroby = '{disease_name}'
    """, engine)

    opis = pd.read_sql(f"""
        SELECT opis FROM choroby
        WHERE nazwa_choroby = '{disease_name}'
    """, engine)

    return leki, diety, zalecenia, opis

def save_recommendation(disease, confidence, algorithm):
    engine = get_engine()
    with engine.connect() as conn:
        query = text("""
            INSERT INTO rekomendacje (kod_pacjenta, przewidziana_choroba_id, pewnosc_predykcji, algorytm)
            VALUES (:kod_pacjenta,
                    (SELECT id FROM choroby WHERE nazwa_choroby = :disease),
                    :confidence,
                    :algorithm)
        """)
        conn.execute(query, {
            "kod_pacjenta": f"PAC-{int(time.time())}",
            "disease": disease,
            "confidence": confidence,
            "algorithm": algorithm
        })
        conn.commit()

# ─────────────────────────────────────────────
# ŁADOWANIE
# ─────────────────────────────────────────────
try:
    res = load_models()
    X_clean, y_clean = load_training_data()
    wagi_dict = load_wagi()
    engine = get_engine()
except Exception as e:
    st.error(f"Błąd ładowania zasobów: {e}")
    st.stop()

# ─────────────────────────────────────────────
# NAWIGACJA
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=80)
st.sidebar.title("AuraMed")
st.sidebar.caption("System Rekomendacji Terapii")
st.sidebar.divider()

zakładka = st.sidebar.radio(
    "Nawigacja",
    ["Diagnostyka", "Statystyki", "Porównanie algorytmów", "Historia"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption(f"Baza danych: **{y_clean.nunique()}** chorób")
st.sidebar.caption(f"Przypadki treningowe: **{len(X_clean)}**")
st.sidebar.caption(f"Liczba objawów: **{len(res['feature_ids'])}**")

# ═══════════════════════════════════════════════
# ZAKŁADKA 1 — DIAGNOSTYKA
# ═══════════════════════════════════════════════
if zakładka == "Diagnostyka":
    st.title("Panel Diagnostyczny")
    st.markdown("Wybierz objawy pacjenta i wygeneruj rekomendację diagnostyczną wraz z planem leczenia.")

    col_form, col_info = st.columns([2, 1])

    with col_form:
        st.subheader("Wybór objawów")

        algorytm = st.selectbox(
            "Algorytm klasyfikacji:",
            [
                "Logistic Regression (najwyższa odporność na błędy)",
                "Random Forest (wysoka dokładność, analiza ważności objawów)"
            ]
        )

        pary_objawow = przygotuj_liste_objawow(res['feature_names'])
        polskie_nazwy = [p[0] for p in pary_objawow]
        angielskie_nazwy = [p[1] for p in pary_objawow]

        wybrane_pl = st.multiselect(
            "Objawy pacjenta:",
            options=polskie_nazwy,
        )

        objawy_wybrane = [angielskie_nazwy[polskie_nazwy.index(name)] for name in wybrane_pl]

        generuj = st.button("Generuj rekomendację", use_container_width=True, type="primary")

    with col_info:
        st.subheader("Instrukcja")
        st.info("""
        1. Wybierz algorytm klasyfikacji
        2. Zaznacz objawy pacjenta z listy
        3. Kliknij **Generuj rekomendację**
        4. System wyświetli diagnozę i plan leczenia
        """)
        st.warning("⚠️ System służy wyłącznie jako wsparcie decyzji. Ostateczna diagnoza należy do lekarza.")

    if generuj:
        if not objawy_wybrane:
            st.error("Wybierz co najmniej jeden objaw.")
        else:
            feature_ids = res['feature_ids']
            input_vector = np.zeros(len(res['feature_ids']))
            for i, name in enumerate(res['feature_names']):
                if name in objawy_wybrane:
                    input_vector[i] = wagi_dict.get(feature_ids[i], 1)

            if "Logistic" in algorytm:
                model = res['lr']
                alg_name = "Logistic Regression"
            else:
                model = res['rf']
                alg_name = "Random Forest"

            probs = model.predict_proba([input_vector])[0]
            classes = model.classes_
            top_idx = np.argsort(probs)[::-1][:5]

            top_disease = classes[top_idx[0]]
            top_prob = probs[top_idx[0]]

            st.divider()

            col_diag, col_prob = st.columns([2, 1])
            with col_diag:
                st.success(f"### Główne rozpoznanie: **{top_disease}**")
            with col_prob:
                st.metric("Pewność modelu", f"{top_prob*100:.1f}%")

            st.subheader("Top 5 najbardziej prawdopodobnych diagnoz")
            df_top = pd.DataFrame({
                'Diagnoza': [classes[i] for i in top_idx],
                'Prawdopodobieństwo': [f"{probs[i]*100:.1f}%" for i in top_idx]
            })
            st.dataframe(df_top, use_container_width=True, hide_index=True)

            # Podobni pacjenci
            st.subheader("👥 Podobni pacjenci z bazy danych")
            sims = cosine_similarity([input_vector], X_clean.values)[0]
            top_sim_idx = np.argsort(sims)[::-1][:5]
            podobni = pd.DataFrame({
                'ID przypadku': X_clean.index[top_sim_idx],
                'Diagnoza': y_clean.iloc[top_sim_idx].values,
                'Podobieństwo': [f"{sims[i]*100:.1f}%" for i in top_sim_idx]
            })
            st.dataframe(podobni, use_container_width=True, hide_index=True)

            # Szczegóły podobnych pacjentów
            st.subheader("🔍 Szczegóły podobnych przypadków")
            for idx, sim_idx in enumerate(top_sim_idx[:3]):
                choroba_podobnego = y_clean.iloc[sim_idx]
                with st.expander(f"Przypadek {idx+1} — {choroba_podobnego} (podobieństwo: {sims[sim_idx]*100:.1f}%)"):
                    objawy_podobnego = [
                        res['feature_names'][i]
                        for i, val in enumerate(X_clean.iloc[sim_idx])
                        if val > 0
                    ]
                    objawy_pl = [tlumacz_objaw(o) for o in objawy_podobnego]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Objawy tego pacjenta:**")
                        for o in objawy_pl:
                            st.markdown(f"- {o}")
                    with col_b:
                        st.write("**Wspólne objawy z nowym pacjentem:**")
                        wspolne = [tlumacz_objaw(o) for o in objawy_podobnego if o in objawy_wybrane]
                        if wspolne:
                            for o in wspolne:
                                st.markdown(f"✅ {o}")
                        else:
                            st.write("Brak dokładnie wspólnych objawów")

            # Plan leczenia
            st.divider()
            st.subheader(f"Plan leczenia dla: {top_disease}")

            leki, diety, zalecenia, opis = get_recommendations(top_disease)

            if not opis.empty and opis.iloc[0]['opis']:
                st.info(f"**Opis choroby:** {tlumacz_chorobe(opis.iloc[0]['opis'])}")

            tab1, tab2, tab3 = st.tabs(["Leki", "Dieta", "Zalecenia"])

            with tab1:
                if not leki.empty:
                    for _, row in leki.iterrows():
                        st.markdown(f"- {row['nazwa_leku']}")
                else:
                    st.info("Brak danych o lekach dla tej choroby.")

            with tab2:
                if not diety.empty:
                    for _, row in diety.iterrows():
                        st.markdown(f"- {tlumacz_diete(row['nazwa_diety'])}")
                else:
                    st.info("Brak danych dietetycznych dla tej choroby.")

            with tab3:
                if not zalecenia.empty:
                    for _, row in zalecenia.iterrows():
                        st.markdown(f"- {tlumacz_zalecenie(row['tresc_zalecenia'])}")
                else:
                    st.info("Brak zaleceń dla tej choroby.")

            try:
                save_recommendation(top_disease, float(top_prob), alg_name)
                st.caption("✅ Rekomendacja zapisana w historii.")
            except:
                st.caption("⚠️ Nie udało się zapisać rekomendacji w historii.")

# ═══════════════════════════════════════════════
# ZAKŁADKA 2 — STATYSTYKI
# ═══════════════════════════════════════════════
elif zakładka == "Statystyki":
    st.title("Statystyki i Eksploracja Danych")

    tab1, tab2, tab3 = st.tabs([
        "Rozkład diagnoz",
        "Najczęstsze zalecenia",
        "Podobieństwo chorób"
    ])

    with tab1:
        st.subheader("Liczba przypadków dla każdej diagnozy")
        dist = y_clean.value_counts().reset_index()
        dist.columns = ['Choroba', 'Liczba przypadków']
        st.bar_chart(dist.set_index('Choroba'))

    with tab2:
        st.subheader("Top 10 najczęstszych zaleceń")
        query_recs = '''
            SELECT z.tresc_zalecenia, COUNT(cz.choroba_id) as liczba_chorob
            FROM zalecenia z
            JOIN choroba_zalecenia cz ON z.id = cz.zalecenie_id
            GROUP BY z.tresc_zalecenia
            ORDER BY liczba_chorob DESC
            LIMIT 10
        '''
        df_recs = pd.read_sql(query_recs, engine)
        st.bar_chart(df_recs.set_index('tresc_zalecenia'))

    with tab3:
        st.subheader("Macierz korelacji objawów (Top 20)")
        query_matrix = '''
            SELECT po.przypadek_id, o.nazwa_objawu, 1 as wystepuje
            FROM przypadek_objawy po
            JOIN objawy o ON po.objaw_id = o.id
        '''
        df_matrix = pd.read_sql(query_matrix, engine)
        pivot = df_matrix.pivot(
            index='przypadek_id',
            columns='nazwa_objawu',
            values='wystepuje'
        ).fillna(0)
        top_20 = df_matrix['nazwa_objawu'].value_counts().nlargest(20).index
        corr = pivot[top_20].corr()
        st.dataframe(corr.round(2), use_container_width=True)

# ═══════════════════════════════════════════════
# ZAKŁADKA 3 — PORÓWNANIE ALGORYTMÓW
# ═══════════════════════════════════════════════
elif zakładka == "Porównanie algorytmów":
    st.title("Porównanie Algorytmów Uczenia Maszynowego")
    st.markdown("Porównanie skuteczności algorytmów na czystych danych oraz przy różnych poziomach szumu.")

    def wprowadz_szum(X, poziom, seed=42):
        np.random.seed(seed)
        X_n = X.copy().astype(float)
        n_flip = int(X_n.size * poziom)
        rows = np.random.randint(0, X_n.shape[0], n_flip)
        cols = np.random.randint(0, X_n.shape[1], n_flip)
        for r, c in zip(rows, cols):
            X_n.iloc[r, c] = 1 - X_n.iloc[r, c]
        return X_n

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
        "k-NN"               : KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes"        : GaussianNB(),
        "SGD"                : SGDClassifier(random_state=42, max_iter=1000),
        "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    poziomy = [0, 0.05, 0.10, 0.20]
    etykiety = ['0%', '5%', '10%', '20%']

    if st.button("Uruchom eksperyment", type="primary"):
        wyniki = {name: [] for name in models_dict}
        progress = st.progress(0)
        status = st.empty()
        total = len(models_dict) * len(poziomy)
        krok = 0

        for poziom in poziomy:
            X_tr_n = wprowadz_szum(X_tr, poziom) if poziom > 0 else X_tr.copy()
            for name, model in models_dict.items():
                status.text(f"Testuję: {name} przy szumie {int(poziom*100)}%...")
                model.fit(X_tr_n, y_tr)
                acc = accuracy_score(y_te, model.predict(X_te))
                wyniki[name].append(acc)
                krok += 1
                progress.progress(krok / total)

        status.text("✅ Eksperyment zakończony!")

        df_wyniki = pd.DataFrame(wyniki, index=etykiety).T
        df_wyniki.columns = ['Szum 0%', 'Szum 5%', 'Szum 10%', 'Szum 20%']
        df_wyniki['Spadek'] = df_wyniki['Szum 0%'] - df_wyniki['Szum 20%']
        df_wyniki = df_wyniki.sort_values('Spadek')

        st.subheader("Wyniki eksperymentu")
        st.dataframe(
            df_wyniki.round(4).style.highlight_min(
                subset=['Spadek'], color='lightgreen'
            ).highlight_max(
                subset=['Spadek'], color='#ffcccc'
            ),
            use_container_width=True
        )

        st.subheader("Wykres — wpływ szumu na dokładność")
        st.line_chart(pd.DataFrame(wyniki, index=etykiety))

        st.success(f"✅ Najbardziej odporny algorytm: **{df_wyniki.index[0]}** (spadek: {df_wyniki.iloc[0]['Spadek']:.4f})")
        st.error(f"❌ Najmniej odporny algorytm: **{df_wyniki.index[-1]}** (spadek: {df_wyniki.iloc[-1]['Spadek']:.4f})")
    else:
        st.info("Kliknij przycisk aby uruchomić eksperyment porównawczy. Może to potrwać około 30 sekund.")

# ═══════════════════════════════════════════════
# ZAKŁADKA 4 — HISTORIA REKOMENDACJI
# ═══════════════════════════════════════════════
elif zakładka == "Historia":
    st.title("📋 Historia Rekomendacji")
    st.markdown("Przeglądaj historię wszystkich wygenerowanych rekomendacji diagnostycznych.")

    query_historia = '''
        SELECT
            r.id,
            r.kod_pacjenta,
            c.nazwa_choroby AS diagnoza,
            ROUND(CAST(r.pewnosc_predykcji * 100 AS numeric), 1) AS pewnosc,
            r.algorytm,
            TO_CHAR(r.created_at, 'YYYY-MM-DD HH24:MI') AS data
        FROM rekomendacje r
        JOIN choroby c ON r.przewidziana_choroba_id = c.id
        ORDER BY r.created_at DESC
    '''

    try:
        df_historia = pd.read_sql(query_historia, engine)

        if df_historia.empty:
            st.info("Brak historii rekomendacji.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Łączna liczba rekomendacji", len(df_historia))
            with col2:
                st.metric("Najczęstsza diagnoza",
                         df_historia['diagnoza'].value_counts().index[0])
            with col3:
                st.metric("Średnia pewność modelu",
                         f"{df_historia['pewnosc'].mean():.1f}%")

            st.divider()

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                choroby_lista = ['Wszystkie'] + sorted(df_historia['diagnoza'].unique().tolist())
                filtr_choroba = st.selectbox("Filtruj po diagnozie:", choroby_lista)
            with col_f2:
                filtr_algorytm = st.selectbox(
                    "Filtruj po algorytmie:",
                    ['Wszystkie', 'Logistic Regression', 'Random Forest']
                )

            df_filtered = df_historia.copy()
            if filtr_choroba != 'Wszystkie':
                df_filtered = df_filtered[df_filtered['diagnoza'] == filtr_choroba]
            if filtr_algorytm != 'Wszystkie':
                df_filtered = df_filtered[df_filtered['algorytm'] == filtr_algorytm]

            st.subheader(f"Znalezione rekomendacje: {len(df_filtered)}")
            st.dataframe(
                df_filtered.rename(columns={
                    'id': 'ID',
                    'kod_pacjenta': 'Kod pacjenta',
                    'diagnoza': 'Diagnoza',
                    'pewnosc': 'Pewność (%)',
                    'algorytm': 'Algorytm',
                    'data': 'Data i godzina'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Rozkład diagnoz w historii")
            dist_historia = df_historia['diagnoza'].value_counts().reset_index()
            dist_historia.columns = ['Diagnoza', 'Liczba rekomendacji']
            st.bar_chart(dist_historia.set_index('Diagnoza'))

    except Exception as e:
        st.error(f"Błąd pobierania historii: {e}")