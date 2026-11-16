"""Inference pipeline reproduced from Fraud Detection V2 notebook.
Never fit preprocessors at inference time.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ARTIFACT_DIR = Path(os.getenv("FRAUDSHIELD_ARTIFACT_DIR", "artifacts"))


class FraudPredictor:
    def __init__(self, directory: Path = ARTIFACT_DIR):
        directory = Path(directory)
        with (directory / "feature_schema.json").open(encoding="utf-8") as file:
            self.schema = json.load(file)
        with (directory / "decision_policy.json").open(encoding="utf-8") as file:
            self.policy = json.load(file)
        self.imputer_missing = joblib.load(directory / "imputer_missing.joblib")
        self.scaler_numeric = joblib.load(directory / "scaler_numeric.joblib")
        self.imputer_balcon = joblib.load(directory / "imputer_balcon.joblib")
        self.scaler_balcon = joblib.load(directory / "scaler_balcon.joblib")
        self.encoder = joblib.load(directory / "encoder.joblib")
        self.model = CatBoostClassifier()
        self.model.load_model(str(directory / "catboost_v2.cbm"))
        assert list(self.model.feature_names_) == self.schema["model_features"], "Model/schema mismatch"

    def transform(self, records: list[dict]) -> pd.DataFrame:
        if not records:
            raise ValueError("At least one record is required")
        required = self.schema["raw_features"]
        for i, record in enumerate(records):
            missing = sorted(set(required) - set(record))
            extra = sorted(set(record) - set(required))
            if missing or extra:
                raise ValueError(f"Record {i}: missing={missing}; unexpected={extra}")
        data = pd.DataFrame(records, columns=required)
        cats = self.schema["categorical_columns"]
        numeric = [c for c in required if c not in cats]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="raise")
            if not np.isfinite(data[col].dropna().to_numpy(dtype=float)).all():
                raise ValueError(f"Invalid non-finite value in {col}")
        if data[numeric].isna().any().any():
            raise ValueError("Raw numeric values cannot be null; use documented sentinel values where applicable")
        if data[cats].isna().any().any():
            raise ValueError("Categorical values cannot be null")
        for col in cats:
            if not data[col].map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"Categorical field {col} must be a string")

        # Exact order used in the corrected notebook, V2 cells 161–168 and 215.
        for col in self.schema["cols_missing"]:
            data.loc[data[col] == -1, col] = np.nan
            data[f"missing_{col}"] = data[col].isna().astype(int)
        missing_cols = self.schema["cols_missing"]
        data[missing_cols] = self.imputer_missing.transform(data[missing_cols])
        numeric_cols = self.schema["cols_scaling"]
        data[numeric_cols] = self.scaler_numeric.transform(data[numeric_cols])

        col = "intended_balcon_amount"
        data.loc[data[col] < 0, col] = np.nan
        data[f"missing_{col}"] = data[col].isna().astype(int)
        data[col] = self.imputer_balcon.transform(data[[col]]).ravel()
        data[col] = self.scaler_balcon.transform(data[[col]]).ravel()

        expected_pre = self.schema["pre_encoding_features"]
        if data.columns.tolist() != expected_pre:
            raise ValueError("Pre-encoding feature names or order differ from training")
        one_hot = pd.DataFrame(self.encoder.transform(data[cats]),
                               columns=self.encoder.get_feature_names_out(), index=data.index)
        final = pd.concat([data.drop(columns=cats), one_hot], axis=1)
        if final.columns.tolist() != self.schema["model_features"]:
            raise ValueError("Model feature names or order differ from training")
        return final

    def predict(self, records: list[dict]) -> list[dict]:
        final = self.transform(records)
        scores = self.model.predict_proba(final)[:, 1]
        threshold = float(self.policy["threshold"])
        return [
            {"fraud_probability": float(score),
             "threshold": threshold,
             "decision": "manual_review" if score >= threshold else "no_alert",
             "model_version": self.policy["model_version"]}
            for score in scores
        ]
