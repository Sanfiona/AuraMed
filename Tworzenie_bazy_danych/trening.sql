-- Dodajemy klucze główne
ALTER TABLE choroby ADD PRIMARY KEY (id);
ALTER TABLE objawy ADD PRIMARY KEY (id);
ALTER TABLE zalecenia ADD PRIMARY KEY (id);
ALTER TABLE diety ADD PRIMARY KEY (id);
-- Tabela dla pojedynczych wierszy z pliku treningowego
CREATE TABLE przypadki_treningowe (
    id SERIAL PRIMARY KEY,
    choroba_id INT REFERENCES choroby(id)
);

-- Tabela łącząca konkretny przypadek z jego objawami (tylko te, które mają wartość 1)
CREATE TABLE przypadek_objawy (
    przypadek_id INT REFERENCES przypadki_treningowe(id),
    objaw_id INT REFERENCES objawy(id)
);

-- Połączenie CHOROBY <-> DIETY
ALTER TABLE choroba_dieta 
    ADD CONSTRAINT fk_dieta_choroba FOREIGN KEY (choroba_id) REFERENCES choroby(id) ON DELETE CASCADE,
    ADD CONSTRAINT fk_dieta_id FOREIGN KEY (dieta_id) REFERENCES diety(id) ON DELETE CASCADE;

-- Połączenie CHOROBY <-> ZALECENIA
ALTER TABLE choroba_zalecenia 
    ADD CONSTRAINT fk_zalecenie_choroba FOREIGN KEY (choroba_id) REFERENCES choroby(id) ON DELETE CASCADE,
    ADD CONSTRAINT fk_zalecenie_id FOREIGN KEY (zalecenie_id) REFERENCES zalecenia(id) ON DELETE CASCADE;

-- Połączenie CHOROBY <-> OBJAWY (z częstotliwością, o której pisaliśmy wcześniej)
ALTER TABLE choroba_objawy 
    ADD CONSTRAINT fk_objawy_choroba FOREIGN KEY (choroba_id) REFERENCES choroby(id) ON DELETE CASCADE,
    ADD CONSTRAINT fk_objawy_id FOREIGN KEY (objaw_id) REFERENCES objawy(id) ON DELETE CASCADE;