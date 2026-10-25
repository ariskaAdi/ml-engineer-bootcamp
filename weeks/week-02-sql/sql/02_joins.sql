-- Week 2 · Bagian 2 — JOIN dan subquery

-- name: inner_join_nama_produk_dan_kota
-- INNER JOIN: hanya baris yang punya pasangan di kedua tabel.
SELECT o.order_id, o.order_date, p.product_name, p.category, c.city_name,
       o.quantity * o.unit_price AS revenue
FROM orders o
JOIN products p ON p.product_id = o.product_id
JOIN cities   c ON c.city_id    = o.city_id
ORDER BY o.order_id
LIMIT 5;

-- name: inner_join_revenue_per_kota
-- Medan TIDAK muncul karena tidak punya order.
SELECT c.city_name, SUM(o.quantity * o.unit_price) AS revenue
FROM cities c
JOIN orders o ON o.city_id = c.city_id
GROUP BY c.city_name
ORDER BY revenue DESC;

-- name: left_join_revenue_per_kota
-- LEFT JOIN: semua baris tabel kiri tetap ada. Kota tanpa order -> NULL, diubah jadi 0 oleh COALESCE.
-- Perhatikan: COUNT(o.order_id) menghitung 0 untuk Medan, sedangkan COUNT(*) akan menghitung 1.
SELECT c.city_name,
       COUNT(o.order_id)                            AS n_orders,
       COALESCE(SUM(o.quantity * o.unit_price), 0)  AS revenue
FROM cities c
LEFT JOIN orders o ON o.city_id = c.city_id
GROUP BY c.city_name
ORDER BY revenue DESC;

-- name: subquery_order_di_atas_rata_rata
-- Subquery di WHERE: bandingkan setiap order dengan rata-rata semua order.
SELECT COUNT(*) AS n_orders_above_avg
FROM orders
WHERE quantity * unit_price > (SELECT AVG(quantity * unit_price) FROM orders);

-- name: cte_produk_di_atas_rata_rata_kategori
-- CTE (WITH ...) = subquery yang diberi nama, lebih mudah dibaca.
WITH product_revenue AS (
    SELECT p.product_name, p.category, SUM(o.quantity * o.unit_price) AS revenue
    FROM orders o JOIN products p USING (product_id)
    GROUP BY p.product_name, p.category
)
SELECT product_name, category, revenue
FROM product_revenue pr
WHERE revenue >= (SELECT AVG(revenue) FROM product_revenue x WHERE x.category = pr.category)
ORDER BY revenue DESC;
