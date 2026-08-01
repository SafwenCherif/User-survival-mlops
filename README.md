# User Survival Prediction — Full MLOps Project Description

**Repository:** [https://github.com/SafwenCherif/User-survival-mlops](https://github.com/SafwenCherif/User-survival-mlops)  
**Project:** Titanic passenger survival prediction end-to-end MLOps pipeline  
**Environment documented here:** Ubuntu (Linux) + Docker + Python 3.12  
**Author:** Safwen Cherif

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [High-level architecture](#2-high-level-architecture)
3. [Technology stack and why each tool](#3-technology-stack-and-why-each-tool)
4. [Project structure (every folder and file)](#4-project-structure-every-folder-and-file)
5. [End-to-end data and ML flow](#5-end-to-end-data-and-ml-flow)
6. [Step-by-step build log (what we did and why)](#6-step-by-step-build-log-what-we-did-and-why)
7. [Deep dive: modules, functions, and integrations](#7-deep-dive-modules-functions-and-integrations)
8. [Ubuntu vs Windows notes](#8-ubuntu-vs-windows-notes)
9. [How to clone and run this project](#9-how-to-clone-and-run-this-project)
10. [Useful day-to-day commands](#10-useful-day-to-day-commands)
11. [Credentials and security](#11-credentials-and-security)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What this project is

This project is a **production-style MLOps system** (kept intentionally simple on the ML side) that:

1. Stores the raw Titanic CSV in a **GCP Cloud Storage bucket**
2. Orchestrates an **ETL DAG** with **Astro (Apache Airflow)** to load that CSV into **PostgreSQL**
3. Ingests data from Postgres into local train/test CSVs with **psycopg2**
4. Explores data and models in a **Jupyter notebook**
5. Builds a lightweight **feature store in Redis**
6. Processes features (imputation, encoding, feature engineering, SMOTE) and **writes features to Redis**
7. Trains a **Random Forest** model by **reading features from Redis**, with hyperparameter tuning
8. Runs a unified **training pipeline** and versions **code + data/model artifacts** on GitHub
9. Serves predictions through a **Flask** web app
10. Detects **input data drift** with **Alibi Detect (KSDrift)**
11. Exposes custom metrics with **Prometheus** and visualizes them in **Grafana**

The goal is not maximum Kaggle score — it is to understand **how ML systems are wired in real life**: ingestion, storage, orchestration, feature serving, training, deployment, drift, and monitoring.

---

## 2. High-level architecture

```text
┌──────────────────┐     Astro Airflow DAG      ┌─────────────────┐
│  GCP Cloud       │  list → download → load    │  PostgreSQL     │
│  Storage bucket  │ ─────────────────────────► │  (Astro)        │
│  Titanic CSV     │   google_cloud_default     │  table: titanic │
└──────────────────┘   + service account key    └────────┬────────┘
                                                         │
                                                         │ psycopg2
                                                         ▼
                                                ┌─────────────────┐
                                                │ Data Ingestion  │
                                                │ train/test CSV  │
                                                │ artifacts/raw/  │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Data Processing │
                                                │ + SMOTE         │
                                                └────────┬────────┘
                                                         │ store features
                                                         ▼
                                                ┌─────────────────┐
                                                │ Redis Feature   │
                                                │ Store           │
                                                └────────┬────────┘
                                                         │ extract features
                                                         ▼
                                                ┌─────────────────┐
                                                │ Model Training  │
                                                │ RandomForest    │
                                                │ artifacts/models│
                                                └────────┬────────┘
                                                         │ load .pkl
                                                         ▼
                     ┌──────────────────────────────────────────────┐
                     │ Flask application.py                         │
                     │  /predict  +  Alibi KSDrift                  │
                     │  /metrics  → Prometheus counters             │
                     └───────────────┬──────────────────────────────┘
                                     │ scrape :5000/metrics
                                     ▼
                            ┌─────────────────┐
                            │ Prometheus      │
                            └────────┬────────┘
                                     │ datasource
                                     ▼
                            ┌─────────────────┐
                            │ Grafana         │
                            │ dashboards      │
                            └─────────────────┘
```

---

## 3. Technology stack and why each tool

| Layer | Tool | Why it is used |
|-------|------|----------------|
| Cloud object storage | **GCP Cloud Storage** | Durable landing zone for the raw CSV; simulates “data arrives in the cloud” |
| Auth to GCP | **Service account + JSON key** | Non-interactive auth for Airflow operators (no human login in DAGs) |
| Orchestration | **Astro CLI + Apache Airflow** | Schedules and monitors ETL tasks; Astro simplifies local Airflow with Docker |
| Warehouse / OLTP store | **PostgreSQL** | Structured store after ETL; source of truth for ingestion into the ML pipeline |
| Python DB driver | **psycopg2 / psycopg2-binary** | Connects the training code to Postgres from the host |
| Packaging | **setuptools / `pip install -e .`** | Installs the project as an editable package so `src.*` and `config.*` imports work |
| Logging | **custom logger (`src/logger.py`)** | Persistent daily log files for debugging pipelines |
| Errors | **CustomException** | Structured error messages with file/line context |
| Exploration | **Jupyter notebook** | Quick EDA and model choice (Random Forest ~0.86 in notebook) |
| Feature store | **Redis** | Fast key-value store for entity features (online-feature-store pattern) |
| Imbalance handling | **imbalanced-learn (SMOTE)** | Balances survival classes before modeling concepts are practiced |
| Modeling | **scikit-learn RandomForest + RandomizedSearchCV** | Simple, strong baseline; easy to pickle and serve |
| Serving | **Flask** | Lightweight prediction UI + HTTP metrics endpoint |
| Drift detection | **Alibi Detect KSDrift** | Statistical test comparing live inputs to reference (Redis) distribution |
| Metrics | **prometheus_client** | Exposes `prediction_count_total` and `drift_count_total` |
| Metrics DB | **Prometheus** | Scrapes and stores time-series metrics |
| Visualization / alerts | **Grafana** | Dashboards on top of Prometheus |
| Versioning | **Git + GitHub** | Code and artifact versioning (`artifacts/raw`, `artifacts/models`) |
| Containers | **Docker** | Runs Airflow stack, Redis, Prometheus, Grafana reproducibly |

---

## 4. Project structure (every folder and file)

```text
Titanic Survival/
├── application.py              # Flask app: predict + drift + /metrics
├── docker-compose.yml          # Prometheus + Grafana (Ubuntu-ready)
├── prometheus.yml              # Prometheus scrape config for Flask
├── Dockerfile                  # Astro runtime image + Google provider
├── packages.txt                # Astro OS packages (from astro dev init)
├── requirements.txt            # Python dependencies for the ML project
├── setup.py                    # Editable package definition
├── README.md                   # Short Astro/README
├── full-project-description.md # This document
├── .gitignore                  # Secrets, venv, Materials, logs, etc.
├── .dockerignore
├── .env                        # Local env (gitignored)
├── airflow_settings.yaml       # Astro Airflow settings (gitignored)
│
├── config/
│   ├── __init__.py
│   ├── database_config.py      # Postgres connection for host-side Python
│   └── paths_config.py         # Paths for raw/processed artifacts
│
├── src/
│   ├── __init__.py
│   ├── logger.py               # Logging setup
│   ├── custom_exception.py     # Custom exception class
│   ├── data_ingestion.py       # Postgres → train/test CSV
│   ├── feature_store.py        # Redis feature store wrapper
│   ├── data_processing.py      # Clean/engineer features → Redis
│   └── model_training.py       # Redis → train RF → pickle
│
├── pipeline/
│   ├── __init__.py
│   └── training_pipeline.py    # Chains ingestion → processing → training
│
├── dags/
│   ├── extract_data_from_gcp.py# Airflow DAG: GCS → Postgres
│   ├── exampledag.py           # Default Astro example DAG
│   └── .airflowignore
│
├── artifacts/
│   ├── raw/
│   │   ├── titanic_train.csv
│   │   └── titanic_test.csv
│   └── models/
│       └── random_forest_model.pkl
│
├── notebook/
│   ├── titanic.ipynb           # EDA + model experiments
│   └── titanic_train.csv       # Optional notebook copy of data
│
├── templates/
│   └── index.html              # Flask UI form
├── static/
│   └── style.css               # Flask UI styles
│
├── include/                    # Mounted into Airflow (gitignored)
│   └── gcp-key.json            # GCP service account key (SECRET)
│
├── logs/                       # Runtime logs (gitignored)
├── .astro/                     # Astro CLI local config (gitignored)
└── venv/                       # Python virtualenv (gitignored)
```

### What is intentionally **not** in Git

| Path | Reason |
|------|--------|
| `include/gcp-key.json` | Secret credentials |
| `venv/` | Reproducible via `requirements.txt` |
| `Materials/` | Local reference material only; not part of the published project |
| `logs/` | Runtime noise |
| `.astro/`, Airflow local DB files | Local orchestration state |
| `.env` | Local environment |

---

## 5. End-to-end data and ML flow

### Phase A — Cloud → Postgres (Airflow)

1. CSV lives in GCS bucket `titanic-survival-predict-mlops` as `Titanic-Dataset.csv`
2. DAG `extract_titanic_data`:
   - `GCSListObjectsOperator` lists bucket objects
   - `GCSToLocalFilesystemOperator` downloads CSV into the worker (`/tmp/...`)
   - `PythonOperator` loads the CSV into Postgres table `titanic` via SQLAlchemy/psycopg2
3. Airflow connections:
   - `google_cloud_default` → keyfile `/usr/local/airflow/include/gcp-key.json`
   - `postgres_default` → Host = Astro Postgres **container name**, port `5432`

### Phase B — Postgres → local artifacts (ingestion)

`DataIngestion` connects to `localhost:5432` (published host port), runs `SELECT * FROM public.titanic`, splits 80/20, writes:

- `artifacts/raw/titanic_train.csv`
- `artifacts/raw/titanic_test.csv`

### Phase C — Feature engineering + Redis

`DataProcessing`:

- Imputes Age/Embarked/Fare
- Encodes Sex/Embarked
- Creates Familysize, Isalone, HasCabin, Title, Pclass_Fare, Age_Fare
- Applies SMOTE on feature matrix (imbalance handling practice)
- Stores each passenger’s features in Redis as `entity:{PassengerId}:features`

### Phase D — Train from feature store

`ModelTraining`:

- Lists all Redis entity IDs
- Splits entities train/test
- Loads feature JSON blobs from Redis into DataFrames
- Runs `RandomizedSearchCV` on RandomForest
- Saves `artifacts/models/random_forest_model.pkl`

### Phase E — Serve + monitor

`application.py`:

- Loads the pickle model
- Fits a `StandardScaler` on historical Redis features; builds `KSDrift` reference
- On `/predict`: scales input, runs drift test, increments counters, returns Survived / Did Not Survive
- Prometheus scrapes `/metrics`; Grafana visualizes counters

---

## 6. Step-by-step build log (what we did and why)

### 6.1 Project bootstrap

**What:** Created folders (`src`, `config`, `pipeline`, `artifacts`, `templates`, `static`, `logs`), `setup.py`, `requirements.txt`, logger, custom exception, path config.  
**Why:** Clean modular layout; editable install makes imports stable.  
**Command:**

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

### 6.2 GCP setup

**What:**

- GCP project `titanic-survival-predict-mlops`
- Bucket with `Titanic-Dataset.csv`
- Service account with Owner / Storage Object Admin / Storage Object Viewer
- JSON key downloaded and later placed in `include/gcp-key.json`
- Bucket IAM principal = that service account

**Why:** Airflow must authenticate to GCS without interactive user login. A dedicated service account + keyfile path inside the container is a clear, reproducible pattern.

### 6.3 Astro Airflow CLI (Ubuntu)

**What:** Installed Astro CLI (user-local binary when `sudo` installer was unavailable), ran `astro dev init` with runtime `12.6.0`, updated Dockerfile:

```dockerfile
FROM quay.io/astronomer/astro-runtime:12.6.0
RUN pip install apache-airflow-providers-google
```

**Why:** Local Airflow with a pinned Astro runtime; Google provider enables GCS operators.

**Ubuntu fix:** Project name in `.astro/config.yaml` must be **lowercase** (`titanic-survival-prediction`). Docker image tags cannot contain uppercase letters.

**Start:**

```bash
export PATH="$HOME/.local/bin:$PATH"
astro dev start
```

UI (newer CLI): `http://titanic-survival.localhost:6563` — `admin` / `admin`.

### 6.4 Airflow connections

| Connection | Purpose | Key fields |
|------------|---------|------------|
| `google_cloud_default` | GCS access | Keyfile Path = `/usr/local/airflow/include/gcp-key.json`, Scope = `https://www.googleapis.com/auth/cloud-platform` |
| `postgres_default` | Load SQL | Host = `titanic-survival-prediction_f9b101-postgres-1`, DB/user/pass = `postgres`, Port = `5432` |

**Why host is the container name for Airflow:** workers talk on the Docker network.  
**Why DBeaver uses `localhost`:** DBeaver runs on the Ubuntu host and uses the published port `5432`.

**Ubuntu tip:** `chmod 644 include/gcp-key.json` so the Airflow `astro` user can read the mounted key.

### 6.5 ETL DAG

File: `dags/extract_data_from_gcp.py`  
DAG id: `extract_titanic_data`  
Result: table `public.titanic` with **891** rows (verified in DBeaver / `psql`).

### 6.6 Data ingestion with psycopg2

Files: `config/database_config.py`, `src/data_ingestion.py`  

**Ubuntu fix:** use `psycopg2-binary` instead of `psycopg2` (no `pg_config` / libpq build tools required).

```bash
python -m src.data_ingestion
```

### 6.7 Jupyter exploration

Notebook validated Random Forest ~**0.86** accuracy. Kept simple on purpose.

### 6.8 Redis feature store

```bash
docker pull redis
docker run -d --name redis-container -p 6379:6379 redis
```

File: `src/feature_store.py` — thin wrapper around Redis `SET`/`GET` of JSON feature dicts.

### 6.9 Data processing + feature storing

```bash
python -m src.data_processing
```

Writes ~712 train passenger feature keys into Redis (plus any extras you must avoid, e.g. smoke-test keys without `Survived`).

### 6.10 Model training from Redis

```bash
python -m src.model_training
```

Observed accuracy ~**0.825** after cleanup of a leftover Redis test key that lacked labels.

### 6.11 Training pipeline + GitHub versioning

```bash
python -m pipeline.training_pipeline
git init
git add ...
git commit ...
git push -u origin main
```

Repo: `https://github.com/SafwenCherif/User-survival-mlops.git`

### 6.12 Flask + Alibi Detect + Prometheus + Grafana

```bash
# monitoring stack (docker compose if available, else docker run — see section 9)
python application.py
```

URLs:

- App: http://localhost:5000  
- Metrics: http://localhost:5000/metrics  
- Prometheus: http://localhost:9090  
- Grafana: http://localhost:3000 (`admin` / `admin`)

Grafana datasource URL (from inside Docker network): `http://prometheus:9090`  
Metrics visualized: `prediction_count_total`, `drift_count_total`.

**Ubuntu fixes:**

1. Prometheus scrape target `host.docker.internal:5000` needs `extra_hosts: host.docker.internal:host-gateway` on Linux.
2. Flask `debug=True` + `start_http_server(8000)` conflicts on reload → `use_reloader=False`.

---

## 7. Deep dive: modules, functions, and integrations

### 7.1 `src/logger.py`

- Creates `logs/` and a daily file `log_YYYY-MM-DD.log`
- `get_logger(name)` returns a module logger at INFO  
**Why:** Pipelines should leave an audit trail outside the terminal.

### 7.2 `src/custom_exception.py`

- Wraps failures with filename + line number  
**Why:** Faster debugging when nested try/except blocks fire.

### 7.3 `config/paths_config.py`

- `RAW_DIR`, `TRAIN_PATH`, `TEST_PATH`, `PROCESSED_DIR`  
**Why:** Single source of truth for artifact locations.

### 7.4 `config/database_config.py`

- Host `localhost` for **host-side** Python talking to published Postgres port  
**Why different from Airflow:** process location (host vs container network) changes the hostname you must use.

### 7.5 `src/data_ingestion.py` — class `DataIngestion`

| Method | What | Why |
|--------|------|-----|
| `connect_to_db` | `psycopg2.connect(**DB_CONFIG)` | Open a DB session |
| `extract_data` | `SELECT * FROM public.titanic` into DataFrame | Pull warehouse data into ML land |
| `save_data` | `train_test_split` + CSV write | Create reproducible local splits |
| `run` | Orchestrates the three steps | Single entrypoint |

### 7.6 `src/feature_store.py` — class `RedisFeatureStore`

| Method | What | Why |
|--------|------|-----|
| `store_features` | `SET entity:{id}:features` JSON | Per-entity feature document |
| `get_features` | `GET` + `json.loads` | Online feature lookup |
| `store_batch_features` / `get_batch_features` | Loop helpers | Bulk load/read |
| `get_all_entity_ids` | `KEYS entity:*:features` | Training set discovery |

**Why Redis:** Sub-millisecond reads, simple mental model for “feature store” before Feast/Vertex.

### 7.7 `src/data_processing.py` — class `DataProcessing`

| Method | What | Why |
|--------|------|-----|
| `load_data` | Read train/test CSV | Input for processing |
| `preprocess_data` | Impute/encode/engineer | Model-ready columns |
| `handle_imbalance_data` | SMOTE on selected features | Address class imbalance |
| `store_feature_in_redis` | Batch write passenger features + label | Persist features for training/serving reference |
| `retrive_feature_redis_store` | Fetch one entity | Sanity check / demo |
| `run` | Full processing pipeline | One command execution |

### 7.8 `src/model_training.py` — class `ModelTraining`

| Method | What | Why |
|--------|------|-----|
| `load_data_from_redis` | Fetch feature dicts for IDs | Train from feature store, not only CSV |
| `prepare_data` | Split entities, build X/y | Avoid leakage pattern practice |
| `hyperparamter_tuning` | `RandomizedSearchCV` | Lightweight HPO |
| `train_and_evaluate` | Fit best model, log accuracy | Report quality |
| `save_model` | Pickle to `artifacts/models/` | Artifact for serving |
| `run` | Full training pipeline | One command execution |

### 7.9 `pipeline/training_pipeline.py`

Chains:

1. `DataIngestion(...).run()`
2. `DataProcessing(...).run()`
3. `ModelTraining(...).run()`

**Why:** One reproducible “train everything” entrypoint for demos and CI later.

### 7.10 `dags/extract_data_from_gcp.py`

- Uses Google transfer/list operators + Python load function  
- Hardcodes Postgres host as the Astro Postgres container name (must match `docker ps`)  
**Why:** Demonstrates cloud extract → warehouse load under orchestration.

### 7.11 GCP bucket + service account (integration rationale)

| Piece | Role |
|-------|------|
| Bucket | Raw immutable landing zone |
| Service account | Machine identity for Airflow |
| JSON key in `include/` | Mounted into container at `/usr/local/airflow/include` via Astro volume config |
| Scope `cloud-platform` | Broad Google API access for development simplicity |

### 7.12 `application.py` (Flask + drift + metrics)

| Piece | Role |
|-------|------|
| Load `random_forest_model.pkl` | Inference |
| `fit_scaler_on_ref_data` | Build reference matrix from Redis features |
| `KSDrift(x_ref=..., p_val=0.05)` | Drift detector |
| `prediction_count` / `drift_count` Counters | Prometheus custom metrics |
| `/` | HTML form |
| `/predict` | Inference + optional drift increment |
| `/metrics` | Prometheus exposition format |
| `start_http_server(8000)` | Extra metrics server on port 8000; Prometheus scraping uses `:5000/metrics` |

### 7.13 Prometheus + Grafana

| File / setting | Role |
|----------------|------|
| `prometheus.yml` | Scrape `flask_app` at `host.docker.internal:5000` every 15s |
| `docker-compose.yml` | Run Prometheus `:9090` and Grafana `:3000` on `monitoring` network |
| Grafana datasource | `http://prometheus:9090` (service DNS name inside Docker) |
| Dashboard metrics | `prediction_count_total`, `drift_count_total` |

**Why this monitoring path:** You see live operational signals (traffic + drift events) without opening application logs.

---

## 8. Ubuntu vs Windows notes

| Topic | Docker Desktop (Windows/Mac) | Ubuntu (this project) |
|-------|------------------------------|------------------------|
| Astro install | winget / Homebrew / exe | `curl install.astronomer.io` or user-local GitHub binary in `~/.local/bin` |
| Docker image name | May appear more forgiving | Project name **must be lowercase** |
| `psycopg2` | Often wheels fine | Prefer **`psycopg2-binary`** |
| Prometheus → host Flask | `host.docker.internal` works via Docker Desktop | Add `extra_hosts: host.docker.internal:host-gateway` |
| Flask debug + metrics port | May work | Use `use_reloader=False` to avoid port `8000` bind clash |
| DBeaver Postgres host | `localhost` | Same (`localhost:5432`) |
| Airflow Postgres host | Container name | Same idea; name is environment-specific |
| Compose CLI | Often bundled | May need `docker-compose` binary or plain `docker run` |

---

## 9. How to clone and run this project

### 9.1 Prerequisites

- Ubuntu (or similar Linux)
- Python 3.12+
- Docker Engine running
- Git
- (Optional) GCP project + bucket + SA key if you want to re-run the Airflow GCS DAG
- (Optional) Astro CLI for Airflow
- (Optional) DBeaver to browse Postgres

### 9.2 Clone and Python env

```bash
git clone https://github.com/SafwenCherif/User-survival-mlops.git
cd User-survival-mlops

python3.12 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

### 9.3 Start Redis

```bash
docker pull redis
docker run -d --name redis-container -p 6379:6379 --restart=no redis
# later: docker start redis-container
```

### 9.4 Start Astro Airflow (ETL / Postgres)

Install Astro CLI if needed, then:

```bash
export PATH="$HOME/.local/bin:$PATH"
astro version
astro dev start
```

Login to the printed Airflow URL (`admin` / `admin`).

Place your GCP key at:

```text
include/gcp-key.json
```

```bash
chmod 644 include/gcp-key.json
```

Create Airflow connections as in section 6.4 (update Postgres **Host** to your actual container name from `docker ps`).

Update bucket name in `dags/extract_data_from_gcp.py` if different, unpause DAG `extract_titanic_data`, trigger it.

Verify:

```bash
docker exec <your-postgres-container> psql -U postgres -d postgres -c "SELECT COUNT(*) FROM titanic;"
```

### 9.5 Run ML pipelines (host)

With Redis + Postgres port published:

```bash
source venv/bin/activate
python -m src.data_ingestion
python -m src.data_processing
python -m src.model_training

# or all-in-one:
python -m pipeline.training_pipeline
```

Check logs under `logs/` and model at `artifacts/models/random_forest_model.pkl`.

### 9.6 Start Prometheus + Grafana

If you have Compose:

```bash
docker compose up -d
# or: docker-compose up -d
```

If Compose is missing, equivalent:

```bash
docker network create monitoring 2>/dev/null || true

docker run -d --name prometheus --network monitoring -p 9090:9090 \
  -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml" \
  --add-host=host.docker.internal:host-gateway \
  --restart=no prom/prometheus:latest

docker run -d --name grafana --network monitoring -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  --restart=no grafana/grafana:latest
```

### 9.7 Start Flask app

```bash
source venv/bin/activate
python application.py
```

Open:

- http://localhost:5000  
- http://localhost:5000/metrics  
- http://localhost:9090  
- http://localhost:3000  

Grafana login: `admin` / `admin`  
Add datasource Prometheus → URL `http://prometheus:9090` → Save & test  
Create dashboard visualizations for `prediction_count_total` and `drift_count_total`.

### 9.8 Minimal path (if you already have artifacts + Redis features)

If `artifacts/` and Redis are already populated (e.g. after cloning plus restoring Redis by re-running processing), you can skip Airflow and only run:

```bash
docker start redis-container
# start prometheus/grafana as above
python application.py
```

To rebuild Redis features from CSVs:

```bash
python -m src.data_processing
```

---

## 10. Useful day-to-day commands

```bash
# Start work
export PATH="$HOME/.local/bin:$PATH"
cd "/path/to/User-survival-mlops"
source venv/bin/activate
astro dev start
docker start redis-container
docker start prometheus grafana   # if created
python application.py

# Stop work (save RAM)
# Ctrl+C Flask
astro dev stop
docker stop redis-container prometheus grafana

# Prevent Docker auto-start on boot
docker update --restart=no $(docker ps -aq)

# Inspect
docker ps
docker exec redis-container redis-cli DBSIZE
curl -s http://localhost:5000/metrics | grep _total
```

---

## 11. Credentials and security

| Secret / credential | Default / location | Notes |
|---------------------|--------------------|-------|
| Airflow UI | `admin` / `admin` | Local only |
| Astro Postgres | `postgres` / `postgres` @ `localhost:5432` | Local only |
| Grafana | `admin` / `admin` | Change for anything non-local |
| GCP SA JSON | `include/gcp-key.json` | **Never commit**; already gitignored |

Treat this as a **local/dev MLOps stack**, not a hardened production deployment (Owner role on the service account is convenient for development but overly privileged for real systems).

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `repository name must be lowercase` on `astro dev start` | Capital letters in Astro project name | Lowercase `project.name` in `.astro/config.yaml` |
| `pg_config executable not found` | Building `psycopg2` from source | Use `psycopg2-binary` |
| GCS tasks fail with permission / file not found | Key not readable in container | Confirm mount + `chmod 644` key |
| DAG load to SQL fails | Wrong Postgres host | Use exact `...-postgres-1` name from `docker ps` |
| `Input y contains NaN` in training | Redis keys without `Survived` | Delete bad keys; re-run processing |
| Prometheus target DOWN on Ubuntu | `host.docker.internal` unresolved | `extra_hosts: host.docker.internal:host-gateway` |
| Flask crash `Address already in use` port 8000 | Debug reloader double-start | `use_reloader=False` |
| Grafana can’t reach Prometheus | Wrong URL | Use `http://prometheus:9090` (not localhost) from Grafana container |
| High RAM after reboot | Containers restart policies | `docker update --restart=no ...` and stop stacks when idle |

---

## Closing summary

You built a complete MLOps learning system:

**GCP → Airflow ETL → PostgreSQL → Python ingestion → Redis feature store → model training → GitHub versioning → Flask serving → Alibi drift detection → Prometheus → Grafana.**

Each piece answers a real production question:

- Where does raw data land? (**GCS**)
- Who moves it reliably? (**Airflow**)
- Where is structured data? (**Postgres**)
- Where are ML features served from? (**Redis**)
- How is the model produced and stored? (**training pipeline + pickle + Git**)
- How do users get predictions? (**Flask**)
- How do we know inputs shifted? (**Alibi Detect**)
- How do we observe the service? (**Prometheus + Grafana**)

That is the “what” and the “why” of the entire project.
