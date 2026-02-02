GlobalBrine — Web Platform Design Document

Purpose

This document describes the design of a lightweight, AI-style web platform for deploying and serving the GlobalBrine-LithiumModel. The platform focuses on global visualization, single-point prediction, and batch prediction, while maintaining strong reproducibility, scientific transparency, and engineering simplicity.

⸻

1. Vision & Target Users

1.1 Vision

Provide an interactive, reproducible, and visually compelling platform that allows users to:
	•	Explore global brine chemistry and irradiance data
	•	Apply a pretrained MAE-based lithium extraction model
	•	Generate predictions for individual sites or large datasets
	•	Support scientific research, techno-economic analysis, and decision-making

1.2 Target Users
	•	Researchers / scientists: exploring latent structure, predictions, and global patterns
	•	Engineers / analysts: evaluating lithium extraction potential of brines
	•	Policy / industry stakeholders: high-level comparison and visualization
	•	ML / data engineers: integrating the model into production workflows

⸻

2. Functional Requirements

2.1 Core Features (MVP)
	1.	Global Data Visualization
	•	Interactive world map showing brine samples and/or model predictions
	•	Multiple layers: raw data, predicted selectivity, crystallization rate, evaporation rate
	•	Filters by geography, chemistry (TDS, MLR), and data source
	2.	Single-Point Prediction
	•	Web form for entering one brine sample (partial inputs allowed)
	•	CPU-based inference using the deployed model
	•	Output of:
	•	Li⁺/Mg²⁺ selectivity
	•	Li crystallization rate
	•	Evaporation rate
	•	Display of model metadata (version, tag, scaler)
	3.	Batch Prediction
	•	Upload CSV or GeoJSON
	•	Asynchronous processing for large jobs
	•	Downloadable prediction results (CSV)
	•	Optional visualization as a map layer
	4.	Reproducibility & Transparency
	•	Display current model version (e.g. v0.2.1)
	•	Link predictions to:
	•	Git tag
	•	CHANGELOG entry
	•	Scaler version
	•	Input hash / job ID
	5.	AI-Style UI
	•	Modern, minimal, research-grade aesthetic
	•	Dark or semi-dark theme with subtle gradients and motion

2.2 Future / Nice-to-Have
	•	User accounts and saved sessions
	•	Export figures and reports (PNG / PDF)
	•	Uncertainty estimation and confidence intervals
	•	Latent-space similarity search
	•	Collaboration and sharing

⸻

3. Non-Functional Requirements
	•	Performance: single inference < 500 ms on CPU
	•	Scalability: horizontal scaling of inference workers
	•	Security: safe file upload, API rate limiting
	•	Maintainability: containerized deployment, CI/CD
	•	Reproducibility: deterministic inference tied to model artifacts

⸻

4. System Architecture

Browser (React)
   ↓
API Gateway (NGINX)
   ↓
FastAPI Backend
   ├── Model Inference Service (CPU)
   ├── Batch Worker (Celery / RQ)
   ├── Redis (queue & cache)
   ├── Object Storage (S3 / MinIO)
   └── Metadata DB (PostgreSQL / SQLite)

Monitoring: Prometheus + Grafana
Logging: Loki / ELK
CI/CD: GitHub Actions

4.1 Recommended Tech Stack
	•	Frontend: React + Vite
	•	Styling: TailwindCSS + shadcn/ui
	•	Maps: Mapbox GL JS or MapLibre
	•	Backend: FastAPI + Uvicorn/Gunicorn
	•	ML Runtime:
	•	PyTorch (eager) or TorchScript / ONNX Runtime (preferred)
	•	Async Jobs: Celery or RQ + Redis
	•	Storage: S3 (production) / MinIO (local)
	•	Database: PostgreSQL (prod) / SQLite (MVP)
	•	Infra: Docker + Kubernetes (or docker-compose for MVP)

⸻

5. API Design (High Level)

5.1 Model Metadata

GET /api/v1/model

Returns:
	•	model version
	•	git tag
	•	checksum
	•	feature schema

5.2 Single Prediction

POST /api/v1/predict

Request (JSON):

{
  "Li_gL": 0.1,
  "Mg_gL": 5.0,
  "Na_gL": 100.0,
  "K_gL": 2.0,
  "Ca_gL": 1.0,
  "SO4_gL": 10.0,
  "Cl_gL": 120.0,
  "MLR": 0.05,
  "TDS_gL": 350.0,
  "Light_kW_m2": 0.18
}

Response:
	•	predictions
	•	normalization info
	•	model metadata

5.3 Batch Prediction
	•	POST /api/v1/predict/batch
	•	GET /api/v1/predict/batch/{job_id}/status
	•	GET /api/v1/predict/batch/{job_id}/result

5.4 Data Access
	•	GET /api/v1/data/points (GeoJSON)
	•	GET /api/v1/data/tiles (optional, raster/vector tiles)

⸻

6. Frontend & UX Design

6.1 Visual Style (AI-Inspired)
	•	Dark background with blue-purple gradients
	•	Glass-morphism cards (blur + translucency)
	•	Subtle animations and transitions
	•	Typography: Inter / IBM Plex Sans

6.2 Key Pages
	1.	Landing / Dashboard
	•	Project overview
	•	Coverage statistics
	•	Quick links to map & prediction
	2.	Global Map Explorer
	•	Layer selector (raw data vs predictions)
	•	Variable switcher
	•	Filters (TDS, MLR, region)
	•	Clickable points with detailed cards
	3.	Single Prediction Page
	•	Input form (partial inputs allowed)
	•	Prediction output with charts
	•	Model/version disclosure
	4.	Batch Prediction Page
	•	File upload + schema hints
	•	Job status & history
	•	Download results
	5.	Model & Reproducibility Page
	•	Changelog
	•	Release tags
	•	Training & inference notes

⸻

7. Model Serving & Data Flow
	1.	Backend loads:
	•	MAE encoder
	•	Regression head
	•	Feature scaler
	2.	Inputs validated via Pydantic schemas
	3.	Missing values allowed and passed as NaN
	4.	Predictions logged with:
	•	input hash
	•	model tag
	•	timestamp

⸻

8. Deployment & CI/CD

8.1 Development
	•	docker-compose with:
	•	frontend
	•	backend
	•	redis
	•	minio

8.2 Production
	•	Container registry (GHCR / Docker Hub)
	•	Kubernetes + Helm
	•	Ingress + TLS
	•	Horizontal scaling for API & workers

8.3 CI/CD Flow
	•	PR → lint + test
	•	Merge to main → build & deploy staging
	•	Git tag vX.Y.Z → production release

⸻

9. Monitoring & Observability
	•	Metrics: latency, throughput, queue size, CPU usage
	•	Logs: structured JSON logs
	•	Alerts: error rate, stalled batch jobs

⸻

10. Security & Compliance
	•	File size/type validation
	•	API rate limiting
	•	CORS & CSP configuration
	•	Secrets via environment variables

⸻

11. Testing Strategy
	•	Unit tests (API, model utils)
	•	Integration tests (end-to-end prediction)
	•	Performance tests (single & batch)
	•	Edge cases (missing / extreme chemistry)

⸻

12. Roadmap (Suggested)

Phase 1 (Weeks 1–2)
	•	UI mockups (Figma)
	•	API contracts
	•	Minimal FastAPI service

Phase 2 (Weeks 3–4)
	•	Map explorer
	•	Single prediction
	•	Model deployment

Phase 3 (Weeks 5–6)
	•	Batch prediction
	•	Reproducibility metadata
	•	CI/CD

Phase 4 (Weeks 7–8)
	•	Hardening, monitoring, docs

⸻

13. Next Steps
	•	Confirm MVP scope
	•	Decide on ONNX vs PyTorch runtime
	•	Prepare model artifacts with versioning
	•	Generate OpenAPI spec and starter repo

⸻

This document defines a clean, extensible foundation for deploying the GlobalBrine lithium prediction model as a modern scientific web platform.
