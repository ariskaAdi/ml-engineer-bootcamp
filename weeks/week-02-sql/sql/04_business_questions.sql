-- Week 2 · Mini-task — 5 pertanyaan bisnis, SQL saja.
-- Jawaban diverifikasi dengan Pandas di 02_business_questions.py

-- name: q1_revenue_per_bulan
SELECT strftime('%Y-%m', order_date) AS month,
       SUM(quantity * unit_price)    AS revenue
FROM orders
GROUP BY month
ORDER BY month;

-- name: q2_produk_teratas_dan_pangsa
SELECT p.product_name,
       SUM(o.quantity * o.unit_price) AS revenue,
       ROUND(100.0 * SUM(o.quantity * o.unit_price)
             / (SELECT SUM(quantity * unit_price) FROM orders), 1) AS share_pct
FROM orders o JOIN products p USING (product_id)
GROUP BY p.product_name
ORDER BY revenue DESC;

-- name: q3_revenue_per_kota_termasuk_yang_kosong
SELECT c.city_name,
       COUNT(o.order_id)                           AS n_orders,
       COALESCE(SUM(o.quantity * o.unit_price), 0) AS revenue
FROM cities c LEFT JOIN orders o USING (city_id)
GROUP BY c.city_name
ORDER BY revenue DESC;

-- name: q4_produk_terlaris_per_kota
WITH ranked AS (
    SELECT c.city_name, p.product_name,
           SUM(o.quantity * o.unit_price) AS revenue,
           ROW_NUMBER() OVER (PARTITION BY c.city_name
                              ORDER BY SUM(o.quantity * o.unit_price) DESC) AS rn
    FROM orders o
    JOIN products p USING (product_id)
    JOIN cities   c USING (city_id)
    GROUP BY c.city_name, p.product_name
)
SELECT city_name, product_name, revenue FROM ranked WHERE rn = 1 ORDER BY city_name;

-- name: q5_kategori_per_bulan_dengan_pangsa
WITH m AS (
    SELECT strftime('%Y-%m', o.order_date) AS month, p.category,
           SUM(o.quantity * o.unit_price) AS revenue
    FROM orders o JOIN products p USING (product_id)
    GROUP BY month, p.category
)
SELECT month, category, revenue,
       ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY month), 1) AS share_in_month_pct
FROM m
ORDER BY month, revenue DESC;
