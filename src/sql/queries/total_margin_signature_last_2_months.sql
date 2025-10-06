WITH anchor AS (
  -- First day of the latest month that exists in transactions
  SELECT date_trunc('month', MAX(date_transaction))::date AS anchor_month
  FROM transactions
),
month_window AS (
  -- Two-month window: previous month (inclusive) through end of anchor month
  SELECT
    (anchor_month - INTERVAL '1 month')::date AS start_date,        -- start of prior month
    (anchor_month + INTERVAL '1 month')::date AS end_exclusive      -- start of month after anchor
  FROM anchor
)
SELECT
  SUM(t.marge_nette_magasin) AS total_margin_last_2_months
FROM transactions t
JOIN product p
  ON p.code_modele_couleur_actuel = t.modele_couleur_ref
JOIN month_window w ON TRUE
WHERE p.signature_product = TRUE
  AND t.date_transaction >= w.start_date
  AND t.date_transaction <  w.end_exclusive;
