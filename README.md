# AI-Based Identification and Sorting of Recyclable Materials

College mini-project that classifies a photo into **cardboard**, **glass**, **metal**, **paper**, **plastic**, or **trash**, then suggests a sorting stream.

The trained TensorFlow/Keras model `waste_sorting_model.keras` is used **as-is**. This repository does not retrain it and does not publish a fabricated accuracy number.

An earlier Streamlit prototype (`app.py`) is still present and unchanged.

## Architecture

- **Frontend:** React + Vite, green sustainability UI, talks to the API through `VITE_API_URL`
- **Backend:** FastAPI + TensorFlow, loads the Keras model once at startup
- **Inference:** RGB image → resize **224×224** → `tf.keras.applications.mobilenet_v2.preprocess_input` → model predict

## Local setup

### 1. Backend

From the project root (the folder that contains `waste_sorting_model.keras`):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows: copy  |  Unix: cp .env.example .env
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: `GET http://127.0.0.1:8000/health`
- Predict: `POST http://127.0.0.1:8000/predict` with form field `file` (JPG or PNG)
- Docs: `http://127.0.0.1:8000/docs`

Environment variables (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins |
| `MODEL_PATH` | Path to `waste_sorting_model.keras` |
| `MAX_FILE_SIZE_MB` | Upload size limit (default 5) |

### 2. Frontend

```bash
cd frontend
copy .env.example .env   # set VITE_API_URL=http://localhost:8000
npm install
npm run dev
```

Open `http://localhost:5173`. Drag in a JPG/PNG, preview it, then identify the material.

## Tests and checks

```bash
# API tests (loads the existing model; needs TensorFlow)
pytest tests/test_api.py -q

# Frontend production build
cd frontend
npm run build
```

What the API tests cover:

- `/health` reports the model is loaded
- missing or non-image uploads are rejected
- a generated PNG and JPEG return a class from the six labels, a confidence value, all six probabilities, and a recommendation

They do **not** assert a specific class for synthetic solid-color images, and they do **not** claim model accuracy.

## API response shape

`POST /predict` returns JSON similar to:

```json
{
  "predicted_class": "plastic",
  "confidence": 0.0,
  "probabilities": {
    "cardboard": 0.0,
    "glass": 0.0,
    "metal": 0.0,
    "paper": 0.0,
    "plastic": 0.0,
    "trash": 0.0
  },
  "recommendation": "…",
  "recyclable": true,
  "low_confidence": true
}
```

Numbers above are placeholders for the schema only. Real values come from the model at request time.

## Deployment

Keep the **backend on Render** and the **frontend on Vercel**. Do not put secrets in the repo. Do not upload credentials.

### Render (backend)

1. Push this repository to GitHub (the `.keras` file must be in the repo or available at `MODEL_PATH`).
2. In Render, create a **Web Service** from the repo, or use `render.yaml`.
3. Settings:
   - **Runtime:** Python
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Python version:** `3.12.x` (`runtime.txt` / `PYTHON_VERSION`). TensorFlow is large; a paid instance is more reliable than the free tier.
4. Environment variables:
   - `MODEL_PATH=waste_sorting_model.keras`
   - `MAX_FILE_SIZE_MB=5`
   - `CORS_ORIGINS=https://YOUR-VERCEL-APP.vercel.app`
5. After the first Vercel URL exists, update `CORS_ORIGINS` (include `https://localhost:5173` only if you still test locally against the hosted API).
6. Confirm `https://YOUR-RENDER-SERVICE.onrender.com/health` returns `"model_loaded": true`.

**Note:** Cold starts on free Render instances can take minutes because TensorFlow and the model must load.

### Vercel (frontend)

1. Import the `frontend` directory as a Vercel project (Root Directory = `frontend`).
2. Framework: Vite.
3. Environment variable:
   - `VITE_API_URL=https://YOUR-RENDER-SERVICE.onrender.com`
     (no trailing slash)
4. Deploy. `frontend/vercel.json` rewrites unknown routes to `index.html`.
5. Add the Vercel origin to Render `CORS_ORIGINS`, then redeploy or restart the API if needed.

`VITE_API_URL` is baked in at **build time**. If you change it, trigger a new Vercel build.

## Original Streamlit app

The Streamlit UI is unchanged:

```bash
streamlit run app.py
```

## Disclaimer

Predictions are visual only. Contaminated, mixed, or locally restricted items may still belong in residual waste even if the model suggests a recycling stream.
