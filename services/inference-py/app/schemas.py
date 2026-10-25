"""Kontrak request/response. Pydantic menolak input yang salah SEBELUM menyentuh model (HTTP 422)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Job = Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
              "self-employed", "services", "student", "technician", "unemployed", "unknown"]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                    "professional.course", "university.degree", "unknown"]
YesNoUnknown = Literal["no", "yes", "unknown"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
Day = Literal["mon", "tue", "wed", "thu", "fri"]


class ClientFeatures(BaseModel):
    """Satu nasabah. Nama field snake_case; dipetakan ke nama kolom dataset di model.py."""

    model_config = ConfigDict(extra="forbid")   # field tak dikenal -> ditolak (mencegah typo diam-diam)

    age: int = Field(ge=17, le=100)
    job: Job
    marital: Marital
    education: Education
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    contact: Literal["cellular", "telephone"]
    month: Month
    day_of_week: Day
    campaign: int = Field(ge=1, le=100, description="jumlah kontak di kampanye ini, termasuk yang sekarang")
    pdays: int = Field(ge=0, le=999, description="hari sejak kontak terakhir; 999 = belum pernah")
    previous: int = Field(ge=0, le=100)
    poutcome: Literal["failure", "nonexistent", "success"]
    emp_var_rate: float = Field(ge=-10, le=10)
    cons_price_idx: float = Field(ge=80, le=110)
    cons_conf_idx: float = Field(ge=-80, le=0)
    euribor3m: float = Field(ge=0, le=10)
    nr_employed: float = Field(ge=4000, le=6000)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[ClientFeatures] = Field(min_length=1)


class Prediction(BaseModel):
    probability: float
    should_call: bool


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    threshold: float
    test_metrics: dict
