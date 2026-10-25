-- Week 2 · Bagian 1 — SELECT, WHERE, ORDER BY, GROUP BY, HAVING
-- Urutan TULIS : SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT
-- Urutan JALAN : FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
-- (Itu sebabnya alias dari SELECT tidak bisa dipakai di WHERE.)

-- name: lihat_5_order_pertama
SELECT * FROM orders ORDER BY order_id LIMIT 5;

-- name: order_besar_di_bulan_juni
-- WHERE menyaring BARIS sebelum dikelompokkan.
SELECT order_id, order_date, quantity, unit_price, quantity * unit_price AS revenue
FROM orders
WHERE order_date BETWEEN '2026-06-01' AND '2026-06-30'
  AND quantity >= 4
ORDER BY revenue DESC
LIMIT 5;

-- name: ringkasan_per_produk
-- Fungsi agregat: COUNT, SUM, AVG, MIN, MAX
SELECT product_id,
       COUNT(*)                         AS n_orders,
       SUM(quantity)                    AS total_qty,
       ROUND(AVG(quantity), 2)          AS avg_qty,
       SUM(quantity * unit_price)       AS revenue
FROM orders
GROUP BY product_id
ORDER BY revenue DESC;

-- name: produk_dengan_lebih_dari_200_order
-- HAVING menyaring GRUP setelah agregasi (WHERE tidak bisa memakai COUNT).
SELECT product_id, COUNT(*) AS n_orders
FROM orders
GROUP BY product_id
HAVING COUNT(*) > 200;
