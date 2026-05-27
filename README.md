# MLOps Phase 5 — Live ML Pipeline

A production-grade MLOps pipeline built from scratch demonstrating
the full lifecycle: train → track → containerize → deploy → monitor.

## Live Demo

- API: https://mlops-phase5.onrender.com
- Docs: https://mlops-phase5.onrender.com/docs

> Note: Hosted on Render free tier — may take 30 seconds to wake up on first request.

## Architecture

Data → pipeline.py → MLflow tracking → model.pkl (DVC)
↓
FastAPI app
↓
Docker container
↓
Render (live API)
↓
Evidently monitoring

## What's Inside

| File | Purpose |
|---|---|
| app.py | FastAPI RAG app with Groq LLaMA 3.3 70B |
| pipeline.py | Full train → log → save → monitor pipeline |
| monitor.py | Evidently AI drift detection |
| generate_data.py | Reference + current dataset generator |
| Dockerfile | Container config for API |
| Dockerfile.streamlit | Container config for Streamlit UI |
| docker-compose.yml | API + Streamlit together |
| .github/workflows/ci.yml | GitHub Actions CI pipeline |
| model.pkl.dvc | DVC tracked model file |

## Results

| What | Result |
|---|---|
| API response latency | ~1-2 sec (Groq inference) |
| CI pipeline | Passing on every push to main |
| Drift detected | 3/4 columns flagged by Evidently |
| MLflow runs logged | 3 experiments tracked |
| Deployment | Live on Render free tier |

> The model in pipeline.py is trained on synthetic data to demonstrate
> MLOps tooling — the value here is the pipeline architecture, not the
> model itself: MLflow tracking, DVC versioning, Evidently drift detection,
> Docker containerization, and GitHub Actions CI/CD.

## Tech Stack

| Tool | Purpose |
|---|---|
| FastAPI + Uvicorn | API serving |
| MLflow 3.12 | Experiment tracking + model registry |
| Evidently AI 0.4.30 | Data drift monitoring |
| DVC | Data and model versioning |
| Docker + docker-compose | Containerization |
| GitHub Actions | CI/CD pipeline |
| Groq LLaMA 3.3 70B | LLM inference |
| Render | Cloud deployment |

## How to Run Locally

### 1. Clone and install

```bash
git clone https://github.com/Git169-hub/mlops-phase5
cd mlops-phase5
pip install -r requirements.txt
pip install evidently==0.4.30
```

### 2. Set environment variable

```bash
export GROQ_API_KEY=your_key_here
```

Get a free key at: https://console.groq.com

### 3. Generate synthetic data

```bash
python generate_data.py
```

### 4. Run full pipeline (train + track + monitor)

```bash
python pipeline.py
```

### 5. View MLflow experiments

```bash
python -m mlflow ui
```

Open http://127.0.0.1:5000

### 6. View drift report

```bash
open monitoring_report.html
```

### 7. Run API locally

```bash
uvicorn app:app --reload --port 10000
```

API docs at: http://127.0.0.1:10000/docs

### 8. Run with Docker

```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| / | GET | Health check |
| /health | GET | Service status |
| /ask | POST | Ask a question via RAG pipeline |

Example request:

```bash
curl -X POST https://mlops-phase5.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

Example response:

```json
{
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval Augmented Generation...",
  "latency_sec": 1.23
}
```

