# How to Run YatraGov AI (Backend & Frontend)

This document provides the exact commands to run the **Backend API (FastAPI)** and the **Frontend Web Dashboard (Streamlit)** using your `railgov` Conda environment.

---

## 1. Start the Frontend Dashboard (Streamlit UI)

To run the interactive web interface (Passenger Portals, Delay Forecasting, and AI Governance Console):

```bash
conda run -n railgov streamlit run dashboard/app.py
```

- **Local URL**: [http://localhost:8501](http://localhost:8501)
- **Features Available**:
  - 🏠 **Home & Overview**: Time-based greeting and quick-action navigation.
  - 🎫 **Ticket Confirmation**: Calibrated probability odds and TreeSHAP explainability drivers.
  - ⏱️ **Train Delay & Platform Congestion**: Delay error margin range, station crowd classifier, and proactive arrival guidance.
  - 🛡️ **AI Governance Console**: Statistical covariate drift (PSI/KS), model version catalog, rollback controls, immutable SQLite audit trail, and autonomous supervisor agent.

---

## 2. Start the Backend API (FastAPI Service)

To run the REST API server for programmatic predictions and governance endpoints:

```bash
conda run -n railgov uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **API Base URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **OpenAPI Redoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 3. Run Both Concurrently in Separate Terminals

Open two terminal tabs in the `railgov-mvp` directory:

### Terminal 1 (Backend):
```bash
conda run -n railgov uvicorn app.main:app --reload --port 8000
```

### Terminal 2 (Frontend):
```bash
conda run -n railgov streamlit run dashboard/app.py
```

---

## 4. Run the Full Test Suite

To verify all model pipelines, governance safeguards, and UI helpers:

```bash
conda run -n railgov pytest -v
```
*(Expected: 41 passing tests)*
