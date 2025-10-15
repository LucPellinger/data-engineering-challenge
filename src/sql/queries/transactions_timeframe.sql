SELECT 
    MAX(date_transaction) AS last_transaction_date,
    MIN(date_transaction) AS first_transaction_date
FROM transactions;