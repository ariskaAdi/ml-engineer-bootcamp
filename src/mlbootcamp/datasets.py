"""Dataset sintetis untuk latihan.

Kenapa sintetis? Supaya semua script bisa jalan offline, hasilnya selalu sama
(pakai seed), dan kita TAHU pola aslinya — jadi bisa mengecek apakah model
menemukan pola yang benar. Setelah paham, ulangi dengan dataset asli.

Dataset asli (download dari internet) ada di `mlbootcamp.real_data`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# Harga rumah (regresi) — Week 7, 9, 10, 11, 12
# ---------------------------------------------------------------------------
def make_house_prices(n: int = 3000, seed: int = 42, with_missing: bool = True) -> pd.DataFrame:
    """Harga rumah (juta Rupiah) di 5 kota, 2022–2025.

    Pola yang sengaja ditanam:
    - harga naik ~6%/tahun (penting untuk Week 10: split berdasarkan waktu)
    - kota berpengaruh besar (perlu one-hot encoding)
    - luas tanah & bangunan berpengaruh paling besar
    - jarak ke pusat kota menurunkan harga
    - beberapa nilai kosong (perlu imputer)
    """
    rng = np.random.default_rng(seed)
    cities = np.array(["Jakarta", "Surabaya", "Bandung", "Yogyakarta", "Medan"])
    city_mult = {"Jakarta": 1.8, "Surabaya": 1.25, "Bandung": 1.15, "Yogyakarta": 1.0, "Medan": 0.95}

    city = rng.choice(cities, size=n, p=[0.3, 0.2, 0.2, 0.15, 0.15])
    sold_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 4 * 365, n), unit="D")
    land_area = np.round(rng.lognormal(mean=np.log(120), sigma=0.45, size=n)).clip(30, 1000)
    building_area = np.round(land_area * rng.uniform(0.6, 1.6, n)).clip(21, 1200)
    bedrooms = np.clip(np.round(building_area / 45 + rng.normal(0, 0.8, n)), 1, 8).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 6).astype(int)
    age_years = rng.integers(0, 40, n)
    distance_km = np.round(rng.gamma(shape=2.0, scale=5.0, size=n), 1)
    has_garage = (rng.random(n) < 0.6).astype(int)
    certificate = rng.choice(["SHM", "HGB"], size=n, p=[0.7, 0.3])

    years = (sold_date - pd.Timestamp("2022-01-01")).days.to_numpy() / 365
    base_per_m2 = 6.0  # juta / m2 tanah (sebelum faktor lain)
    price = (
        base_per_m2 * land_area
        + 3.5 * building_area
        + 40 * bathrooms
        + 80 * has_garage
        - 4 * age_years
    )
    price = price * np.array([city_mult[c] for c in city])
    price = price * np.exp(-0.02 * distance_km)
    price = price * (1.06 ** years)                      # tren naik per tahun
    price = price * np.where(certificate == "SHM", 1.05, 1.0)
    price = price * rng.lognormal(0, 0.12, n)             # noise
    price = np.round(price.clip(150), 1)

    df = pd.DataFrame({
        "house_id": np.arange(1, n + 1),
        "sold_date": sold_date,
        "city": city,
        "land_area": land_area,
        "building_area": building_area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years.astype(float),
        "distance_km": distance_km,
        "has_garage": has_garage,
        "certificate": certificate,
        "price": price,  # target, juta Rupiah
    })
    if with_missing:
        for col, frac in [("building_area", 0.05), ("age_years", 0.04), ("bathrooms", 0.03)]:
            idx = rng.choice(n, size=int(n * frac), replace=False)
            df[col] = df[col].astype(float)
            df.loc[idx, col] = np.nan
    return df.sort_values("sold_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Customer churn (klasifikasi biner) — Week 8, 9, 15, 16
# ---------------------------------------------------------------------------
def make_churn(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Pelanggan internet/telco. Target `churn` = 1 kalau berhenti langganan.

    Pola: kontrak bulanan, tenure pendek, tagihan tinggi, dan banyak komplain
    -> lebih mungkin churn. Mirip dataset Telco Customer Churn di Kaggle.
    """
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.2])
    tenure = np.where(
        contract == "Month-to-month", rng.integers(0, 40, n),
        np.where(contract == "One year", rng.integers(6, 60, n), rng.integers(12, 72, n)),
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.2])
    base = np.select([internet == "Fiber optic", internet == "DSL"], [80, 55], 25)
    has_tech_support = ((rng.random(n) < 0.35) & (internet != "No")).astype(int)
    monthly = np.round(base + rng.normal(0, 12, n) + 8 * has_tech_support, 2).clip(18, 120)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)
    payment = rng.choice(["Electronic check", "Bank transfer", "Credit card", "Mailed check"],
                         size=n, p=[0.35, 0.22, 0.22, 0.21])
    support_calls = rng.poisson(1.2, n) + (internet == "Fiber optic") * rng.poisson(0.6, n)
    senior = (rng.random(n) < 0.16).astype(int)
    paperless = (rng.random(n) < 0.6).astype(int)

    z = (
        -1.9
        + 1.5 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        - 0.045 * tenure
        + 0.022 * (monthly - 65)
        + 0.6 * (internet == "Fiber optic")
        - 0.7 * has_tech_support
        + 0.5 * (payment == "Electronic check")
        + 0.35 * support_calls
        + 0.3 * senior
        + 0.2 * paperless
    )
    churn = (rng.random(n) < _sigmoid(z)).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "senior_citizen": senior,
        "tenure_months": tenure,
        "contract": contract,
        "internet_service": internet,
        "has_tech_support": has_tech_support,
        "paperless_billing": paperless,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total.astype(float),
        "support_calls": support_calls,
        "churn": churn,
    })
    # Seperti dataset Telco asli: pelanggan baru (tenure 0) total_charges-nya kosong
    df.loc[df["tenure_months"] == 0, "total_charges"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Pelanggan toko (clustering) — Week 14
# ---------------------------------------------------------------------------
def make_customers(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Perilaku belanja pelanggan. Ada kolom `true_segment` yang DISEMBUNYIKAN
    dari model — hanya dipakai untuk mengecek hasil clustering."""
    rng = np.random.default_rng(seed)
    segments = {
        # name: (annual_spend juta, visits/bulan, avg_basket ribu, days_since_last, pct_online)
        "loyal_big": (25, 8, 260, 5, 0.3),
        "occasional": (4, 1.5, 220, 45, 0.2),
        "bargain_frequent": (8, 10, 65, 4, 0.1),
        "online_only": (9, 3, 230, 12, 0.95),
    }
    names = rng.choice(list(segments), size=n, p=[0.2, 0.35, 0.25, 0.2])
    rows = []
    for name in names:
        spend, visits, basket, recency, online = segments[name]
        rows.append({
            "annual_spend": max(0.5, rng.normal(spend, spend * 0.25)),
            "visits_per_month": max(0.2, rng.normal(visits, visits * 0.3)),
            "avg_basket": max(10, rng.normal(basket, basket * 0.2)),
            "days_since_last": max(0, rng.normal(recency, recency * 0.5 + 2)),
            "pct_online": float(np.clip(rng.normal(online, 0.08), 0, 1)),
            "true_segment": name,
        })
    df = pd.DataFrame(rows).round(2)
    df.insert(0, "customer_id", [f"K{i:04d}" for i in range(1, n + 1)])
    return df


# ---------------------------------------------------------------------------
# Transaksi fraud (data sangat tidak seimbang) — Week 15
# ---------------------------------------------------------------------------
def make_fraud(n: int = 20000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Transaksi kartu. Target `is_fraud` hanya ~2% -> akurasi menyesatkan."""
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    amount = np.where(is_fraud, rng.lognormal(6.2, 1.0, n), rng.lognormal(5.0, 0.9, n))
    hour = np.where(is_fraud, rng.choice(24, n, p=_hour_probs(fraud=True)),
                    rng.choice(24, n, p=_hour_probs(fraud=False)))
    km_from_home = np.where(is_fraud, rng.gamma(2, 60, n), rng.gamma(1.5, 8, n))
    new_merchant = np.where(is_fraud, rng.random(n) < 0.6, rng.random(n) < 0.15).astype(int)
    tx_last_24h = np.where(is_fraud, rng.poisson(5, n), rng.poisson(2, n))
    online = np.where(is_fraud, rng.random(n) < 0.7, rng.random(n) < 0.4).astype(int)
    # Sebagian fraud dibuat "mirip normal" supaya tidak terlalu mudah
    stealth = (is_fraud == 1) & (rng.random(n) < 0.3)
    amount[stealth] = rng.lognormal(5.0, 0.9, stealth.sum())
    km_from_home[stealth] = rng.gamma(1.5, 8, stealth.sum())
    return pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "km_from_home": km_from_home.round(1),
        "new_merchant": new_merchant,
        "tx_last_24h": tx_last_24h,
        "online": online,
        "is_fraud": is_fraud,
    })


def _hour_probs(fraud: bool) -> np.ndarray:
    hours = np.arange(24)
    if fraud:
        w = 1 + 3 * ((hours <= 4) | (hours >= 23))
    else:
        w = np.exp(-((hours - 14) ** 2) / 30) + 0.05
    return w / w.sum()
