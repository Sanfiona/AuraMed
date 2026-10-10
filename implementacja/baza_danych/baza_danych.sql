-- AuraMed3 - Schemat bazy danych

-- 1. TABELE SŁOWNIKOWE
CREATE TABLE IF NOT EXISTS choroby (
    id            SERIAL PRIMARY KEY,
    nazwa_choroby VARCHAR(255) UNIQUE NOT NULL,
    opis          TEXT
);

CREATE TABLE IF NOT EXISTS objawy (
    id            SERIAL PRIMARY KEY,
    nazwa_objawu  VARCHAR(255) UNIQUE NOT NULL,
    waga          INT DEFAULT 1
);

CREATE TABLE IF NOT EXISTS leki (
    id         SERIAL PRIMARY KEY,
    nazwa_leku VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS diety (
    id          SERIAL PRIMARY KEY,
    nazwa_diety VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS zalecenia (
    id              SERIAL PRIMARY KEY,
    tresc_zalecenia TEXT UNIQUE NOT NULL
);

-- 2. TABELE RELACYJNE
CREATE TABLE IF NOT EXISTS choroba_objawy (
    choroba_id    INT REFERENCES choroby(id) ON DELETE CASCADE,
    objaw_id      INT REFERENCES objawy(id) ON DELETE CASCADE,
    czestotliwosc FLOAT DEFAULT 1.0,
    PRIMARY KEY (choroba_id, objaw_id)
);

CREATE TABLE IF NOT EXISTS choroba_lek (
    choroba_id INT REFERENCES choroby(id) ON DELETE CASCADE,
    lek_id     INT REFERENCES leki(id) ON DELETE CASCADE,
    PRIMARY KEY (choroba_id, lek_id)
);

CREATE TABLE IF NOT EXISTS choroba_dieta (
    choroba_id INT REFERENCES choroby(id) ON DELETE CASCADE,
    dieta_id   INT REFERENCES diety(id) ON DELETE CASCADE,
    PRIMARY KEY (choroba_id, dieta_id)
);

CREATE TABLE IF NOT EXISTS choroba_zalecenia (
    choroba_id   INT REFERENCES choroby(id) ON DELETE CASCADE,
    zalecenie_id INT REFERENCES zalecenia(id) ON DELETE CASCADE,
    PRIMARY KEY (choroba_id, zalecenie_id)
);

-- 3. TABELE TRENINGOWE
CREATE TABLE IF NOT EXISTS przypadki_treningowe (
    id         SERIAL PRIMARY KEY,
    choroba_id INT REFERENCES choroby(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS przypadek_objawy (
    przypadek_id INT REFERENCES przypadki_treningowe(id) ON DELETE CASCADE,
    objaw_id     INT REFERENCES objawy(id) ON DELETE CASCADE,
    PRIMARY KEY (przypadek_id, objaw_id)
);

-- 4. HISTORIA REKOMENDACJI
CREATE TABLE IF NOT EXISTS rekomendacje (
    id                      SERIAL PRIMARY KEY,
    kod_pacjenta            VARCHAR(20) NOT NULL,
    przewidziana_choroba_id INT REFERENCES choroby(id),
    pewnosc_predykcji       FLOAT,
    algorytm                VARCHAR(50),
    created_at              TIMESTAMP DEFAULT NOW()
);