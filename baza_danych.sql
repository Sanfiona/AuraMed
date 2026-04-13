-- 1. Tabele Słownikowe (Muszą być pierwsze)
CREATE TABLE choroby (
    id INT PRIMARY KEY,
    nazwa_choroby VARCHAR(255) UNIQUE,
    opis TEXT
);

CREATE TABLE objawy (
    id INT PRIMARY KEY,
    nazwa_objawu VARCHAR(255) UNIQUE,
    waga INT
);

CREATE TABLE diety (
    id INT PRIMARY KEY,
    nazwa_diety VARCHAR(255) UNIQUE
);

CREATE TABLE zalecenia (
    id INT PRIMARY KEY,
    tresc_zalecenia TEXT UNIQUE
);

-- 2. Tabele Relacyjne (Łączące)
CREATE TABLE choroba_objawy (
    choroba_id INT REFERENCES choroby(id) ON DELETE CASCADE,
    objaw_id INT REFERENCES objawy(id) ON DELETE CASCADE,
    czestotliwosc FLOAT,
    PRIMARY KEY (choroba_id, objaw_id)
);

CREATE TABLE choroba_dieta (
    choroba_id INT REFERENCES choroby(id) ON DELETE CASCADE,
    dieta_id INT REFERENCES diety(id) ON DELETE CASCADE,
    PRIMARY KEY (choroba_id, dieta_id)
);

CREATE TABLE choroba_zalecenia (
    choroba_id INT REFERENCES choroby(id) ON DELETE CASCADE,
    zalecenie_id INT REFERENCES zalecenia(id) ON DELETE CASCADE,
    PRIMARY KEY (choroba_id, zalecenie_id)
);

-- 3. Tabele Treningowe
CREATE TABLE przypadki_treningowe (
    id SERIAL PRIMARY KEY,
    choroba_id INT REFERENCES choroby(id) ON DELETE CASCADE
);

CREATE TABLE przypadek_objawy (
    przypadek_id INT REFERENCES przypadki_treningowe(id) ON DELETE CASCADE,
    objaw_id INT REFERENCES objawy(id) ON DELETE CASCADE,
    PRIMARY KEY (przypadek_id, objaw_id)
);