SELECT
  COALESCE(SUM(t.marge_nette_magasin), 0) AS total_margin_last_60_days
FROM transactions AS t
JOIN product AS p
  ON p.code_modele_couleur_actuel = t.modele_couleur_ref
WHERE
  p.signature_product IS TRUE
  AND t.date_transaction BETWEEN (
        (SELECT MAX(date_transaction) FROM transactions) - INTERVAL '60 days'
      )
      AND (SELECT MAX(date_transaction) FROM transactions);