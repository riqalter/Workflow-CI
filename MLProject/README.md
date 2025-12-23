# Titanic Classification - MLflow Project

> Machine Learning model untuk Titanic survival prediction dengan MLflow tracking dan GitHub Actions CI/CD.

## 📁 Struktur Repository

```
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI workflow
├── modelling.py            # Training script
├── pyproject.toml          # Python dependencies
├── MLProject               # MLflow Project config
├── titanic_preprocessing/
│   └── train_processed.csv # Dataset
└── README.md
```

## 🚀 Quick Start

### Jalankan Lokal

```bash
# Install dependencies dengan uv
uv pip install .

# Atau dengan pip
pip install -e .

# Jalankan MLflow Project
mlflow run . --env-manager=local
```

### Lihat Hasil

```bash
# MLflow UI lokal
mlflow ui --backend-store-uri file:./mlruns

# Buka browser: http://localhost:5000
```

## ⚙️ GitHub Actions CI

Workflow CI berjalan otomatis saat:

- Push ke branch `main` atau `master`
- Pull request ke branch `main` atau `master`
- Manual trigger via workflow_dispatch

### Level Workflow

| Level        | Steps                                               |
| ------------ | --------------------------------------------------- |
| **Basic**    | Checkout → Setup Python → Install deps → Run MLflow |
| **Skilled**  | + Set tracking URI → Upload artifacts ke GitHub     |
| **Advanced** | + Build Docker → Push ke Docker Hub                 |

## 🔐 Secrets (untuk Advanced)

Tambahkan secrets di GitHub repository settings:

| Secret Name          | Value                   |
| -------------------- | ----------------------- |
| `DOCKERHUB_USERNAME` | Username Docker Hub     |
| `DOCKERHUB_TOKEN`    | Access Token Docker Hub |

## 📊 Artifacts yang Dihasilkan

- `model/` - Model MLflow (MLmodel, model.pkl, dll)
- `estimator.html` - HTML representation model
- `metric_info.json` - Metrics dan parameter
- `training_confusion_matrix.png` - Confusion matrix
- `classification_report.json` - Classification report
- `feature_importance.png` - Feature importance plot

## 🐳 Docker Hub

**Image**: `riqalter/titanic-classifier:latest`

```bash
# Pull dan jalankan
docker pull riqalter/titanic-classifier:latest
docker run -p 5001:8080 riqalter/titanic-classifier:latest
```
