# YatraGov AI
> **PREDICTIVE TRAVEL & AI LIFECYCLE GOVERNANCE**

YatraGov AI is an autonomous AI governance and passenger prediction system designed for Indian Railways travel contexts.

---

### Core Architecture & Capabilities
1. **Passenger Prediction Services**:
   - **Ticket Confirmation Predictor**: Calibrated confirmation odds for Waitlist (WL) and RAC tickets with TreeSHAP log-odds explanations.
   - **Train Delay & Station Congestion**: GradientBoosting delay minutes regressor with holdout error margin range, station platform crowd classification, and proactive advisory guidance.
   - *Roadmap*: Fare Prediction, Fraud Shield, Support Chatbot, Train Recommender, and Journey Planner Agent.

2. **Autonomous AI Governance & Operations Console**:
   - **Real-time Statistical Drift**: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests across baseline and production data streams.
   - **Model Version Registry**: Central catalog tracking model artifacts, versions, and one-click rollback.
   - **Immutable Audit Trail**: SQLite database protected by SQL abort triggers preventing UPDATE or DELETE actions.
   - **Autonomous Supervisor Agent**: Telemetry monitoring, automated candidate retraining on unseen holdouts, and incident escalation.

---

### Running the Dashboard
```bash
conda run -n railgov streamlit run dashboard/app.py
```

---

*Academic prototype. Uses synthetic demo data. Not affiliated with Indian Railways or IRCTC.*