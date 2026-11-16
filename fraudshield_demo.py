"""Bounded, synthetic BAF-like records and transparent economic illustrations.

Not representative sampling. No genuine applicant data is created or required.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def generate_records(count: int, seed: int = 42, profile: str = "mixed") -> list[dict[str, Any]]:
    """Generate self-consistent *illustrative* BAF records (not dataset draws).

    profile: 'routine', 'watchlist', 'mixed'. Neither profile is a ground-truth fraud label.
    """
    if not 1 <= count <= 2500:
        raise ValueError("count must be between 1 and 2500")
    if profile not in ("routine", "watchlist", "mixed"):
        raise ValueError("Unknown generation profile")
    rng = np.random.default_rng(seed)
    result = []

    for _ in range(count):
        unusual = profile == "watchlist" or (profile == "mixed" and rng.random() < 0.35)
        income = float(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        age = int(rng.choice([20, 30, 40, 50, 60, 70, 80], p=[0.09, 0.27, 0.29, 0.20, 0.09, 0.045, 0.015]))
        similarity = float(np.round(rng.beta(1.4 if unusual else 2.3, 4.0 if unusual else 2.0), 6))
        credit_score = int(np.clip(rng.normal(190 if unusual else 105, 75), -170, 385))
        prev = -1 if rng.random() < (0.75 if unusual else 0.60) else int(rng.integers(0, 365))
        curr = int(np.clip(rng.normal(96 if unusual else 145, 72), 0, 390))
        bank = -1 if rng.random() < 0.16 else int(rng.integers(0, 33))
        session = -1.0 if rng.random() < 0.015 else float(np.round(np.clip(rng.gamma(2.0, 5.0), 0.02, 85.0), 4))
        device_emails = -1 if rng.random() < 0.06 else int(rng.choice([0, 1, 2], p=[0.12, 0.77, 0.11]))
        bal = -1.0 if rng.random() < 0.18 else float(np.round(np.clip(rng.normal(26, 28), 0, 112), 4))
        record = {
            "income": income,
            "name_email_similarity": similarity,
            "prev_address_months_count": prev,
            "current_address_months_count": curr,
            "customer_age": age,
            "days_since_request": float(np.round(rng.exponential(1.2 if unusual else 0.7), 6)),
            "intended_balcon_amount": bal,
            "payment_type": str(rng.choice(["AA", "AB", "AC", "AD", "AE"])),
            "zip_count_4w": int(np.clip(rng.gamma(3.2, 300), 1, 6200)),
            "velocity_6h": float(np.round(np.clip(rng.normal(3900, 1700), -140, 17000), 5)),
            "velocity_24h": float(np.round(np.clip(rng.normal(4400, 1400), 1300, 9600), 5)),
            "velocity_4w": float(np.round(np.clip(rng.normal(3800, 650), 2700, 7100), 5)),
            "bank_branch_count_8w": int(np.clip(rng.exponential(120), 0, 2600)),
            "date_of_birth_distinct_emails_4w": int(np.clip(rng.poisson(5 if unusual else 3), 0, 40)),
            "employment_status": str(rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], p=[0.34, 0.20, 0.17, 0.09, 0.09, 0.07, 0.04])),
            "credit_risk_score": credit_score,
            "email_is_free": int(rng.random() < 0.8),
            "housing_status": str(rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], p=[0.47 if unusual else 0.24, 0.11 if unusual else 0.19, 0.13 if unusual else 0.18, 0.09, 0.08, 0.07, 0.05 if unusual else 0.15])),
            "phone_home_valid": int(rng.random() < (0.35 if unusual else 0.72)),
            "phone_mobile_valid": int(rng.random() < 0.86),
            "bank_months_count": bank,
            "has_other_cards": int(rng.random() < (0.72 if unusual else 0.25)),
            "proposed_credit_limit": int(rng.choice([200, 500, 990, 1000, 1500, 2000])),
            "foreign_request": int(rng.random() < (0.20 if unusual else 0.04)),
            "source": str(rng.choice(["INTERNET", "TELEAPP"], p=[0.91, 0.09])),
            "session_length_in_minutes": session,
            "device_os": str(rng.choice(["linux", "macintosh", "other", "windows", "x11"], p=[0.17, 0.08, 0.17, 0.53, 0.05])),
            "keep_alive_session": int(rng.random() < (0.30 if unusual else 0.70)),
            "device_distinct_emails_8w": device_emails,
            "device_fraud_count": 0,
            "month": int(rng.integers(0, 8)),
        }
        result.append(record)
    return result


def expected_decision_costs(scores, threshold: float, cost_fp: float, cost_fn: float) -> dict[str, float]:
    """Model-score-weighted cost proxy, NOT observed FP/FN cost or validated calibration."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("scores must be a finite one-dimensional array of values in [0, 1]")
    if not 0 <= threshold <= 1 or cost_fp < 0 or cost_fn < 0:
        raise ValueError("Invalid decision policy")
    review = values >= threshold
    false_alarm_proxy = float(((1.0 - values) * cost_fp)[review].sum())
    missed_proxy = float((values * cost_fn)[~review].sum())
    return {"review_count": int(review.sum()), "no_alert_count": int((~review).sum()),
            "expected_fp_cost": false_alarm_proxy, "expected_fn_cost": missed_proxy,
            "expected_total_cost": false_alarm_proxy + missed_proxy,
            "no_alert_baseline_proxy": float((values * cost_fn).sum())}


def observed_confusion_costs(labels, scores, threshold: float, cost_fp: float, cost_fn: float) -> dict[str, float]:
    """Observed confusion/cost only when *actual labels* are explicitly supplied."""
    truth = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if truth.shape != scores.shape or truth.ndim != 1 or not np.isin(truth, [0, 1]).all():
        raise ValueError("labels must be aligned binary values")
    if not np.isfinite(scores).all() or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("Invalid scores")
    review = scores >= threshold
    fp = int(((truth == 0) & review).sum())
    fn = int(((truth == 1) & ~review).sum())
    tp = int(((truth == 1) & review).sum())
    tn = int(((truth == 0) & ~review).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "observed_fp_cost": fp * cost_fp, "observed_fn_cost": fn * cost_fn,
            "observed_total_cost": fp * cost_fp + fn * cost_fn,
            "no_alert_baseline": int(truth.sum()) * cost_fn}
