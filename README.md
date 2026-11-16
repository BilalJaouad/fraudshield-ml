# FraudShield — FastAPI starter

Local API for educational scoring of **synthetic** BAF account applications. Do not use for real banking decisions.

## 1. Set up on Windows (VS Code terminal / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` instead of `python` and `.venv\Scripts\pip.exe` instead of `pip`.

## 2. Install Kaggle exports

Unzip your downloaded `fraudshield_artifacts_v2.zip`. Copy the **ten files inside `artifacts/`** into this project's `artifacts/` folder. Do not copy a nested `artifacts/artifacts` folder. Check that `artifacts/catboost_v2.cbm` exists.

**Compatibility:** install the same Python, CatBoost, numpy, pandas and scikit-learn versions shown in `artifacts/model_metadata.json` if loading joblib fails or predictions differ. Joblib can execute code when loaded: only use trusted Kaggle-exported files.

## 3. Run

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs then `GET /health` and `GET /model/info`.

## 4. Test a prediction with a RAW observation

If you have the original Kaggle `Base.csv`, place it somewhere locally, then run, using its actual path:

```powershell
python scripts/make_sample_request.py --csv "C:\\path\\to\\Base.csv" --row 0
```

Open `sample_request.json`, copy its contents into `/docs` -> `POST /predict` -> Try it out -> Execute.

The request structure is `{"features": { ... all 31 raw fields ... }}`. The model currently requires the dataset-relative `month` field and should only be demonstrated with synthetic BAF-compatible records. `model/info` lists the precise raw feature names.

## 5. Caveats

* The notebook's `reference_predictions.csv` contains **preprocessed** inputs. It verifies the CatBoost file but is not a valid raw API request.
* The preserved V2 model includes `month` and `device_fraud_count`. A subsequent model version should reconsider these features; do not invent a value for a real future month.
* The final notebook's exported `model_metadata.json` records library versions. Match them if you have compatibility problems.
* Exposing this demo publicly will require authentication, abuse protection, and non-sensitive demo data.

## 6. Run with Docker (next phase)

First copy the **trusted** Kaggle-exported artifacts into `artifacts/`.
Make sure `artifacts/catboost_v2.cbm` and all five `.joblib` files exist.

Install Docker Desktop (Linux container mode), open the project folder in PowerShell, stop the existing local Uvicorn (`Ctrl+C`), then run:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs --tail=60 api
```

Open http://127.0.0.1:8000/docs and execute `GET /health` and `POST /predict` with the exact same payload previously used locally. The score should match your earlier API response (approximately 0.006042407127395275 for the illustrated test) within floating-point tolerance.

If you get a port error, stop the earlier Uvicorn process. Stop the container with:

```powershell
docker compose down
```

**Compatibility:** `requirements.txt` pins `scikit-learn==1.6.1`, as required for the Kaggle-exported joblib objects. For best reproducibility, also pin numpy/pandas/CatBoost to the versions in the exported `model_metadata.json` after reviewing them. Use only trusted `.joblib` artifacts. The service is designed for synthetic demos and has no authentication; do not expose publicly without security controls.
