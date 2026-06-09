# Run and Deploy Nathra on Vercel

This repo now has a Vercel-friendly shape:

- `index.html`, `src/app.js`, `src/styles.css`: static frontend
- `api/predict.py`: Vercel Python serverless function
- `predictor.py`: shared validation and model prediction logic
- `model/model.pkl` and `model/scaler.pkl`: required trained artifacts, not currently committed

## Local Run

Start the local server:

```bash
python3 local_server.py
```

Open:

```text
http://127.0.0.1:3000
```

The UI will load even before the model exists. Predictions return
`model_not_configured` until these files are added:

```text
model/model.pkl
model/scaler.pkl
```

## Export The Model

The current repo does not include the training CSV or trained pickle files. The
notebook references old Windows paths such as `C:/Users/.../trainTest.csv`, so
you need to bring that CSV into this repo first.

## Generate A Demo Model

If you only want the app to run end to end while you work on the real dataset,
generate synthetic demo artifacts:

```bash
python3.12 scripts/generate_demo_model.py
```

This creates:

```text
model/model.pkl
model/scaler.pkl
model/demo_model.json
```

Do not use this model for medical claims. It is trained on synthetic data and is
only useful for UI, API, and deployment testing.

If you have `trainTest.csv`, run:

```bash
python3.12 scripts/export_model.py --csv path/to/trainTest.csv
```

That creates:

```text
model/model.pkl
model/scaler.pkl
```

Alternatively, run the notebook until the final Random Forest model and scaler
are trained, then save them into the repo's `model/` directory:

```python
from pathlib import Path
import pickle

Path("model").mkdir(exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

Important: export the model using the same Python/scikit-learn version you plan
to run in production. This Vercel setup uses Python 3.12 and
`scikit-learn==1.5.2`.

## API Contract

Health/metadata:

```bash
curl http://127.0.0.1:3000/api/predict
```

Prediction:

```bash
curl -X POST http://127.0.0.1:3000/api/predict \
  -H "content-type: application/json" \
  -d '{
    "age": 50,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 0,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

## Deploy To Vercel

1. Push this repo to GitHub.
2. Import the repo in Vercel.
3. Keep the default framework setting as static/other if Vercel does not detect one.
4. Deploy.

Vercel's Python runtime is currently beta. It detects Python apps from supported
entrypoints and reads dependencies from `requirements.txt`; it also supports
Python 3.12, 3.13, and 3.14. The Python function bundle limit is 500 MB
uncompressed, so keep runtime dependencies tight.

## Production Gaps

This is a deployable portfolio/demo app, not a medical product. Before treating
it as production software, you should add:

- A reproducible training script instead of notebook-only training
- A checked model evaluation report with dataset provenance
- Model versioning and a documented export process
- Server-side logging and monitoring
- Stronger medical disclaimers and review by a domain expert
- Tests for validation, artifact loading, and prediction response shape
