SELECT
  t.point_de_vente,
  COUNT(DISTINCT t.numero_transaction) AS number_distinct_of_transactions,
  COUNT(t.numero_transaction) AS number_of_transactions
FROM transactions t
GROUP BY t.point_de_vente
ORDER BY number_of_transactions DESC;