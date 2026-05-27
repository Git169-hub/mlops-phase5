# MLOps Phase 5 — Live ML Pipeline

A production-grade MLOps pipeline built from scratch.

## Live Demo
- API: https://mlops-phase5.onrender.com
- Docs: https://mlops-phase5.onrender.com/docs

## Architecture
## What's Inside
## Results

| What | Result |
|---|---|
| API response latency | ~1-2 sec (Groq inference) |
| CI pipeline | Passing on every push to main |
| Drift detected | 3/4 columns flagged by Evidently |
| MLflow runs logged | 3 experiments tracked |
| Deployment | Live on Render free tier |

> Note: Model in pipeline.py is trained on synthetic data 
> to demonstrate MLOps tooling — MLflow tracking, DVC versioning,
> Evidently drift detection, Docker containerization, 
> and GitHub Actions CI/CD.

## How to Run Locally

### 1. Install dependencies
`ash
pip install -r requirements.txt
pip install evidently==0.4.30
`

### 2. Generate data
`ash
python generate_data.py
`

### 3. Run full pipeline
`ash
python pipeline.py
`

### 4. View drift report
`ash
start monitoring_report.html
`

### 5. View MLflow UI
`ash
python -m mlflow ui
`
Open http://127.0.0.1:5000

### 6. Run API locally
`ash
uvicorn app:app --reload --port 10000
`

## Results
- Model accuracy: 70%
- Drift detected in 3/4 columns (query_length, response_time_ms, prediction)
- CI pipeline: passing on every push to main
- Live API: deployed on Render free tier

## Tech Stack
- FastAPI + Uvicorn
- MLflow 3.12
- Evidently AI 0.4.30
- DVC
- Docker
- GitHub Actions
- Groq LLM (LLaMA3)
- Render (deployment)
