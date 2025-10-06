-- sql/schema.sql
SET search_path = public;

CREATE TABLE IF NOT EXISTS product (
  code_modele_couleur_actuel TEXT PRIMARY KEY,
  signature_product BOOLEAN
);

CREATE TABLE IF NOT EXISTS transactions (
  transaction_id BIGSERIAL PRIMARY KEY,

  point_de_vente TEXT,
  numero_tpv TEXT,
  numero_transaction TEXT,
  date_transaction DATE,
  heure TIME,
  typologie_magasin TEXT,
  numero_fidelite TEXT,
  type_de_vente TEXT,
  univers_produit TEXT,
  segment_produit TEXT,
  famille_produit TEXT,
  sous_famille_produit TEXT,
  fedas_numero TEXT,
  fedas_libelle TEXT,
  cible_genre_age TEXT,
  modele_couleur_ref TEXT,
  modele_couleur_libelle TEXT,
  type_de_vente_nps TEXT,
  quantite_vendue INTEGER,
  ca_net_ttc NUMERIC,
  ca_net_ht NUMERIC,
  marge_nette_magasin NUMERIC,

  CONSTRAINT fk_tx_modele_ref
    FOREIGN KEY (modele_couleur_ref)
    REFERENCES product (code_modele_couleur_actuel)
    ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_tx_date      ON transactions (date_transaction);
CREATE INDEX IF NOT EXISTS ix_tx_pdv       ON transactions (point_de_vente);
CREATE INDEX IF NOT EXISTS ix_tx_fk_modele ON transactions (modele_couleur_ref);