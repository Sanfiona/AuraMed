"""
test_system.py - Testy jednostkowe systemu AuraMed
Uruchom: pytest test_system.py -v
"""
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────
# FUNKCJA SZUMU (kopia z noise_experiment.py)
# ─────────────────────────────────────────────
def wprowadz_szum(X, poziom_szumu, seed=42):
    np.random.seed(seed)
    X_noisy = X.copy().astype(float)
    n_total = X_noisy.size
    n_flip = int(n_total * poziom_szumu)
    rows = np.random.randint(0, X_noisy.shape[0], n_flip)
    cols = np.random.randint(0, X_noisy.shape[1], n_flip)
    for r, c in zip(rows, cols):
        X_noisy.iloc[r, c] = 1 - X_noisy.iloc[r, c]
    return X_noisy


# ─────────────────────────────────────────────
# DANE TESTOWE (syntetyczne, bez bazy)
# ─────────────────────────────────────────────
@pytest.fixture
def dane_binarne():
    """Prosty binarny zbiór danych do testów."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randint(0, 2, size=(50, 10)),
        columns=[f'objaw_{i}' for i in range(10)]
    )
    y = pd.Series(['choroba_A'] * 25 + ['choroba_B'] * 25)
    return X, y


@pytest.fixture
def modele_wytrenowane(dane_binarne):
    """Wytrenowane modele LR i RF na danych syntetycznych."""
    X, y = dane_binarne
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    lr.fit(X, y)
    rf.fit(X, y)
    return lr, rf, X, y


# ═══════════════════════════════════════════════
# TESTY 1 — FUNKCJA WPROWADZANIA SZUMU
# ═══════════════════════════════════════════════
class TestNoise:

    def test_szum_zmienia_dane(self, dane_binarne):
        """Sprawdza że szum faktycznie zmienia wartości w danych."""
        X, _ = dane_binarne
        X_noisy = wprowadz_szum(X, 0.1)
        assert not X.equals(X_noisy), "Dane po szumie powinny być różne od oryginalnych"

    def test_szum_zero_nie_zmienia(self, dane_binarne):
        """Sprawdza że szum 0% nie zmienia danych."""
        X, _ = dane_binarne
        X_noisy = wprowadz_szum(X, 0.0)
        assert X.equals(X_noisy.astype(X.dtypes)), "Szum 0% nie powinien zmieniać danych"

    def test_szum_zachowuje_ksztalt(self, dane_binarne):
        """Sprawdza że szum nie zmienia kształtu macierzy."""
        X, _ = dane_binarne
        X_noisy = wprowadz_szum(X, 0.2)
        assert X.shape == X_noisy.shape, "Kształt danych po szumie powinien być taki sam"

    def test_szum_wartosci_binarne(self, dane_binarne):
        """Sprawdza że po szumie wartości nadal są 0 lub 1."""
        X, _ = dane_binarne
        X_noisy = wprowadz_szum(X, 0.15)
        unikalne = set(X_noisy.values.flatten())
        assert unikalne.issubset({0.0, 1.0}), "Wartości po szumie powinny być tylko 0 lub 1"

    def test_wiecej_szumu_wiecej_zmian(self, dane_binarne):
        """Sprawdza że wyższy poziom szumu powoduje więcej zmian."""
        X, _ = dane_binarne
        X_5 = wprowadz_szum(X, 0.05, seed=1)
        X_20 = wprowadz_szum(X, 0.20, seed=1)
        zmiany_5 = (X != X_5.astype(int)).sum().sum()
        zmiany_20 = (X != X_20.astype(int)).sum().sum()
        assert zmiany_20 > zmiany_5, "Szum 20% powinien powodować więcej zmian niż 5%"

    def test_szum_powtarzalny(self, dane_binarne):
        """Sprawdza że ten sam seed daje te same wyniki."""
        X, _ = dane_binarne
        X_1 = wprowadz_szum(X, 0.1, seed=42)
        X_2 = wprowadz_szum(X, 0.1, seed=42)
        assert X_1.equals(X_2), "Ten sam seed powinien dawać identyczne wyniki"


# ═══════════════════════════════════════════════
# TESTY 2 — PREDYKCJA MODELI ML
# ═══════════════════════════════════════════════
class TestModele:

    def test_lr_zwraca_predykcje(self, modele_wytrenowane):
        """Sprawdza że Logistic Regression zwraca predykcje."""
        lr, _, X, _ = modele_wytrenowane
        pred = lr.predict(X)
        assert len(pred) == len(X), "Liczba predykcji powinna być równa liczbie przypadków"

    def test_rf_zwraca_predykcje(self, modele_wytrenowane):
        """Sprawdza że Random Forest zwraca predykcje."""
        _, rf, X, _ = modele_wytrenowane
        pred = rf.predict(X)
        assert len(pred) == len(X), "Liczba predykcji powinna być równa liczbie przypadków"

    def test_lr_zwraca_prawdopodobienstwa(self, modele_wytrenowane):
        """Sprawdza że LR zwraca prawdopodobieństwa sumujące się do 1."""
        lr, _, X, _ = modele_wytrenowane
        probs = lr.predict_proba(X)
        sumy = probs.sum(axis=1)
        np.testing.assert_allclose(sumy, 1.0, atol=1e-6,
            err_msg="Prawdopodobieństwa powinny sumować się do 1")

    def test_rf_zwraca_prawdopodobienstwa(self, modele_wytrenowane):
        """Sprawdza że RF zwraca prawdopodobieństwa sumujące się do 1."""
        _, rf, X, _ = modele_wytrenowane
        probs = rf.predict_proba(X)
        sumy = probs.sum(axis=1)
        np.testing.assert_allclose(sumy, 1.0, atol=1e-6,
            err_msg="Prawdopodobieństwa powinny sumować się do 1")

    def test_predykcje_sa_znane_klasy(self, modele_wytrenowane):
        """Sprawdza że predykcje należą do znanych klas."""
        lr, _, X, y = modele_wytrenowane
        pred = lr.predict(X)
        znane_klasy = set(y.unique())
        for p in pred:
            assert p in znane_klasy, f"Predykcja '{p}' nie należy do znanych klas"

    def test_szum_obniza_dokladnosc(self, modele_wytrenowane):
        """Sprawdza że szum obniża dokładność modelu."""
        lr, _, X, y = modele_wytrenowane
        from sklearn.metrics import accuracy_score

        acc_czyste = accuracy_score(y, lr.predict(X))
        X_noisy = wprowadz_szum(X, 0.3)

        lr_noisy = LogisticRegression(max_iter=1000, random_state=42)
        lr_noisy.fit(X_noisy, y)
        acc_szum = accuracy_score(y, lr_noisy.predict(X))

        assert acc_szum <= acc_czyste, \
            "Model trenowany na zaszumionych danych powinien mieć niższą dokładność"

    def test_model_ma_dobre_klasy(self, modele_wytrenowane):
        """Sprawdza że model zna wszystkie klasy ze zbioru treningowego."""
        lr, _, _, y = modele_wytrenowane
        assert set(lr.classes_) == set(y.unique()), \
            "Model powinien znać wszystkie klasy ze zbioru treningowego"


# ═══════════════════════════════════════════════
# TESTY 3 — COSINE SIMILARITY
# ═══════════════════════════════════════════════
class TestCosineSimilarity:

    def test_identyczne_wektory_maja_podobienstwo_1(self):
        """Sprawdza że identyczne wektory mają podobieństwo 1.0."""
        v = np.array([[1, 0, 1, 1, 0]])
        sim = cosine_similarity(v, v)[0][0]
        assert abs(sim - 1.0) < 1e-6, "Identyczne wektory powinny mieć podobieństwo 1.0"

    def test_przeciwne_wektory_maja_podobienstwo_0(self):
        """Sprawdza że ortogonalne wektory mają podobieństwo 0."""
        v1 = np.array([[1, 0, 0]])
        v2 = np.array([[0, 1, 0]])
        sim = cosine_similarity(v1, v2)[0][0]
        assert abs(sim - 0.0) < 1e-6, "Ortogonalne wektory powinny mieć podobieństwo 0"

    def test_podobienstwo_w_przedziale_0_1(self):
        """Sprawdza że podobieństwo jest zawsze w przedziale [0, 1] dla wektorów binarnych."""
        np.random.seed(42)
        X = np.random.randint(0, 2, size=(20, 10)).astype(float)
        for i in range(len(X)):
            for j in range(len(X)):
                sim = cosine_similarity([X[i]], [X[j]])[0][0]
                assert -1e-7 <= sim <= 1.0 + 1e-7, \
                    f"Podobieństwo powinno być w [0,1], otrzymano {sim}"

    def test_znajdowanie_najblizszych_sasiadow(self):
        """Sprawdza że system poprawnie znajduje najbardziej podobnych pacjentów."""
        baza = np.array([
            [1, 1, 0, 0],  # przypadek 0 — bardzo podobny do zapytania
            [0, 0, 1, 1],  # przypadek 1 — bardzo różny
            [1, 0, 0, 0],  # przypadek 2 — częściowo podobny
        ], dtype=float)

        zapytanie = np.array([[1, 1, 0, 0]])  # identyczny jak przypadek 0
        sims = cosine_similarity(zapytanie, baza)[0]
        najblizszy = np.argmax(sims)

        assert najblizszy == 0, "Przypadek 0 powinien być najbardziej podobny"

    def test_wagi_wplywaja_na_podobienstwo(self):
        """Sprawdza że wagi objawów wpływają na wynik podobieństwa."""
        v_bez_wag = np.array([[1, 1, 0, 0]], dtype=float)
        v_z_wagami = np.array([[3, 1, 0, 0]], dtype=float)  # objaw 0 ma wagę 3

        baza = np.array([[3, 1, 0, 0]], dtype=float)

        sim_bez = cosine_similarity(v_bez_wag, baza)[0][0]
        sim_z = cosine_similarity(v_z_wagami, baza)[0][0]

        assert sim_z > sim_bez, \
            "Wektor z wagami powinien być bardziej podobny do bazy z wagami"

    def test_symetria_podobienstwa(self):
        """Sprawdza że podobieństwo jest symetryczne: sim(A,B) == sim(B,A)."""
        v1 = np.array([[1, 0, 1, 0]])
        v2 = np.array([[0, 1, 1, 0]])
        sim_ab = cosine_similarity(v1, v2)[0][0]
        sim_ba = cosine_similarity(v2, v1)[0][0]
        assert abs(sim_ab - sim_ba) < 1e-9, "Podobieństwo powinno być symetryczne"


# ═══════════════════════════════════════════════
# TESTY 4 — PRZYGOTOWANIE WEKTORA WEJŚCIOWEGO
# ═══════════════════════════════════════════════
class TestWektorWejsciowy:

    def test_wektor_ma_poprawna_dlugosc(self):
        """Sprawdza że wektor wejściowy ma długość równą liczbie objawów."""
        feature_names = [f'objaw_{i}' for i in range(132)]
        objawy_wybrane = ['objaw_0', 'objaw_5', 'objaw_10']

        input_vector = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            if name in objawy_wybrane:
                input_vector[i] = 1

        assert len(input_vector) == 132, "Wektor powinien mieć 132 elementy"

    def test_wektor_ma_jedynki_dla_wybranych(self):
        """Sprawdza że wybrane objawy mają wartość > 0 w wektorze."""
        feature_names = ['kaszel', 'goraczka', 'bol_glowy', 'nudnosci']
        objawy_wybrane = ['kaszel', 'goraczka']

        input_vector = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            if name in objawy_wybrane:
                input_vector[i] = 1

        assert input_vector[0] == 1, "Kaszel powinien mieć wartość 1"
        assert input_vector[1] == 1, "Gorączka powinna mieć wartość 1"
        assert input_vector[2] == 0, "Ból głowy powinien mieć wartość 0"
        assert input_vector[3] == 0, "Nudności powinny mieć wartość 0"

    def test_pusty_wybor_daje_zerowy_wektor(self):
        """Sprawdza że brak wybranych objawów daje zerowy wektor."""
        feature_names = [f'objaw_{i}' for i in range(10)]
        objawy_wybrane = []

        input_vector = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            if name in objawy_wybrane:
                input_vector[i] = 1

        assert input_vector.sum() == 0, "Zerowy wybór powinien dać zerowy wektor"

    def test_wszystkie_objawy_daja_pelny_wektor(self):
        """Sprawdza że wybór wszystkich objawów daje wektor samych jedynek."""
        feature_names = [f'objaw_{i}' for i in range(10)]
        objawy_wybrane = feature_names.copy()

        input_vector = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            if name in objawy_wybrane:
                input_vector[i] = 1

        assert input_vector.sum() == 10, "Pełny wybór powinien dać wektor samych jedynek"
