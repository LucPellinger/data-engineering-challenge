SELECT
  t.modele_couleur_ref,
  SUM(t.quantite_vendue) AS total_units_sold
FROM transactions t
GROUP BY t.modele_couleur_ref
ORDER BY total_units_sold DESC
LIMIT 10;