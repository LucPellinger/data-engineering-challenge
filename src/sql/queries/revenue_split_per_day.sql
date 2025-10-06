SELECT
  t.date_transaction AS date,
  SUM(t.ca_net_ttc) AS total_revenue
FROM transactions t
GROUP BY t.date_transaction
ORDER BY t.date_transaction;
