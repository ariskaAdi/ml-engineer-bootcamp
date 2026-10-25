-- Week 2 · Bagian 3 — Window functions
-- GROUP BY  : banyak baris -> SATU baris per grup.
-- Window    : jumlah baris TETAP; setiap baris mendapat nilai hasil hitung dari "jendela" baris lain.
-- Bentuk    : fungsi(...) OVER (PARTITION BY ... ORDER BY ...)

-- name: row_number_vs_rank
-- ROW_NUMBER selalu 1,2,3,... ; RANK memberi nilai sama untuk seri lalu MELOMPAT (1,2,2,4);
-- DENSE_RANK tidak melompat (1,2,2,3).
WITH scores(name, score) AS (
    VALUES ('Ani', 90), ('Budi', 85), ('Citra', 85), ('Dedi', 80)
)
SELECT name, score,
       ROW_NUMBER() OVER (ORDER BY score DESC) AS row_num,
       RANK()       OVER (ORDER BY score DESC) AS rnk,
       DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rnk
FROM scores;

-- name: ranking_produk_di_setiap_kota
-- PARTITION BY = "ulangi penomoran untuk setiap kota".
SELECT c.city_name, p.product_name,
       SUM(o.quantity * o.unit_price) AS revenue,
       RANK() OVER (PARTITION BY c.city_name ORDER BY SUM(o.quantity * o.unit_price) DESC) AS rank_in_city
FROM orders o JOIN products p USING (product_id) JOIN cities c USING (city_id)
GROUP BY c.city_name, p.product_name
ORDER BY c.city_name, rank_in_city
LIMIT 10;

-- name: running_total_revenue_harian
WITH daily AS (
    SELECT order_date, SUM(quantity * unit_price) AS revenue
    FROM orders GROUP BY order_date
)
SELECT order_date, revenue,
       SUM(revenue) OVER (ORDER BY order_date) AS running_total
FROM daily
ORDER BY order_date
LIMIT 7;

-- name: lag_pertumbuhan_bulanan
-- LAG(x) = nilai x di baris SEBELUMNYA (menurut ORDER BY).
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month, SUM(quantity * unit_price) AS revenue
    FROM orders GROUP BY month
)
SELECT month, revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS growth_pct
FROM monthly;

-- name: rata_rata_bergerak_7_hari
WITH daily AS (
    SELECT order_date, SUM(quantity * unit_price) AS revenue FROM orders GROUP BY order_date
)
SELECT order_date, revenue,
       ROUND(AVG(revenue) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0) AS ma7
FROM daily
ORDER BY order_date
LIMIT 10;
