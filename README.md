# FraudShield — Fraud Risk Scoring with CatBoost

Machine learning project based on the **Bank Account Fraud Dataset Suite (NeurIPS 2022), Base variant**. The repository documents the complete ML workflow: exploration, cleaning, preprocessing, model comparison, probability calibration, cost-aware threshold selection, and retrospective evaluation. It also contains an API/Docker prototype and two Streamlit demonstrations.

> **Scope:** Educational portfolio demonstration using **synthetic** banking data. Risk flags are review recommendations, not proof of fraud. Business costs are illustrative, not realized bank losses.

## Live demos

- **Premium dashboard:** https://fraudshield-ml-6t9rpwn6pi49sr9dsmhzxs.streamlit.app/
- **Original demo:** https://fraudshield-ml-idhrbyfndcyogoubdrkggf.streamlit.app/

## Dataset

**Bank Account Fraud Dataset Suite (NeurIPS 2022), variant: `Base`**

https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

The dataset **is not bundled** with the repository. Download it from Kaggle under the dataset's own terms. Locate the Base CSV (commonly `Base.csv`) and put it in a local `data/` folder, which should remain untracked by Git.

```text
fraudshield-ml/
├── data/                         # local only; do not commit
│   └── Base.csv
├── notebooks/
│   └── fraud_detection_experiments.ipynb
├── docs/
│   └── fraudshield_ml_report.pdf
├── artifacts/                    # exported model + fitted transformers
├── api/
│   ├── main.py
│   └── inference.py
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── compose.yaml
```

The exact notebook filename in your checkout may differ. Adjust the notebook's initial CSV path cell to point to `data/Base.csv` (or to the Kaggle dataset path when running in Kaggle). **The existing notebook assumes an explicitly configured path; there is no guarantee it will run unchanged after download.**

## Local setup

Use **Python 3.11**, especially when loading the exported scikit-learn transformers.

```powershell
# Windows PowerShell, from the repository root
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ensure `requirements.txt` pins **`scikit-learn==1.6.1`**. The serialized preprocessing objects were fitted using that version; newer versions may fail to load them or transform data correctly.

### Explore/reproduce the ML experiments

1. Download the Kaggle dataset and select **`Base.csv`**.
2. Place it at `data/Base.csv`, or attach it to a Kaggle notebook session.
3. Open the notebook under `notebooks/`, update its loading path, and **run cells in order**.
4. The workflow uses months 0–4 for model fitting, month 5 for optional calibration, month 6 for threshold selection, and month 7 for final reported evaluation.
5. Model exports are stored in `artifacts/`; do not refit preprocessing using validation or test data.

The full training notebook may also include earlier V1 experiments that use a different temporal split. Those validation scores should not be directly compared with V2 as if they used identical training windows.

### Run Streamlit locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://localhost:8501. The app expects the exported `artifacts/` files (CatBoost `.cbm`, imputer/scaler `.joblib` files, encoder, schema and policy JSON). If artifacts are absent, first export them from the notebook.

### Run the FastAPI prototype locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
```

Open http://127.0.0.1:8000/docs. The key endpoints are `GET /health`, `GET /model/info`, and `POST /predict`. Send the **31 raw BAF input features**, not the final 58 numerical features; the preprocessing is performed server-side.

### Run using Docker

```powershell
docker compose up --build -d
docker compose ps
docker compose down
```

The Docker stack is an alternative to the Streamlit-only deployment. If port 8000 is occupied by a local Uvicorn process, stop it first.

## Methods and test results

| Item | Value |
|---|---:|
| Selected model | CatBoost V2, 330 trees |
| Input → model features | 31 → 58 |
| Test Average Precision | 0.219819 |
| Test ROC–AUC | 0.896774 |
| Test recall (fraud) | 0.39706 |
| Test precision (fraud) | 0.22983 |
| Decision threshold | 0.06029703 |
| Hypothetical FP / FN costs | 100 / 1,000 MAD |
| Hypothetical test cost reduction vs no alerts | 26.4% |

See [the ML technical report](docs/FraudShield_ML_Report.pdf) for preprocessing details, the comparison of model families, probability calibration results, thresholding methodology, confusion matrix, limitations, and evaluation discussion. The test data were accessed during earlier exploratory work, so the reported test evaluation should not be described as perfectly untouched.

## Repository hygiene

Keep `.env` files, credentials, `data/`, downloaded CSVs, local virtual environments, and personally identifying files out of Git. Serialized `joblib` files should only be loaded from trusted sources. Check notebooks for saved outputs or tokens before publishing them.
