"""Preprocessing harga rumah yang dibangun di Week 9, dipakai ulang Week 10–12."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

HOUSE_NUM = ["land_area", "building_area", "bedrooms", "bathrooms", "age_years",
             "distance_km", "has_garage", "years_since_2022", "building_to_land",
             "log_land", "land_x_jakarta"]
HOUSE_CAT = ["city", "certificate", "age_group"]


def add_house_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["years_since_2022"] = (pd.to_datetime(X["sold_date"]) - pd.Timestamp("2022-01-01")).dt.days / 365
    X["building_to_land"] = X["building_area"] / X["land_area"]
    X["log_land"] = np.log(X["land_area"])
    X["age_group"] = pd.cut(X["age_years"], [-1, 5, 15, 30, 100],
                            labels=["baru", "muda", "sedang", "tua"]).astype(str)
    X.loc[X["age_years"].isna(), "age_group"] = np.nan
    X["land_x_jakarta"] = X["land_area"] * (X["city"] == "Jakarta")
    return X


def make_preprocessor(num_cols=HOUSE_NUM, cat_cols=HOUSE_CAT, scale: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols), ("cat", categorical, cat_cols)])


def house_pipeline(model, scale: bool = True) -> Pipeline:
    """feature engineering -> preprocessing -> model"""
    return Pipeline([
        ("features", FunctionTransformer(add_house_features)),
        ("prep", make_preprocessor(scale=scale)),
        ("model", model),
    ])
