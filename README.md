# 🚆 YatraGov AI (RailGov MVP)
> **Autonomous Passenger Intelligence & Responsible AI Governance Platform for Indian Railways**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-black?style=flat)](https://shap.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/Tests-63%20Passed-10B981?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Summary

**YatraGov AI** is an enterprise-grade AI passenger intelligence and governance system engineered for Indian Railways travel contexts (IRCTC/CRIS). 

It bridges the gap between **passenger experience** (demystifying waitlists, predicting delays, finding affordable trains, and multilingual support) and **enterprise governance** (detecting covariate data drift, preventing bot scalping on Tatkal tickets, automated retraining, and tamper-proof immutable audit logging).

---

## 🌟 The 6 Core AI Features

The platform provides 6 purpose-built, accessible AI modules tailored for everyday passengers and railway operations:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           YATRAGOV AI PLATFORM                            │
├─────────────────────────┬───────────────────────┬─────────────────────────┤
│ 1. Ticket Confirmation  │ 2. Train Recommender  │ 3. Delay & Congestion   │
│    Waitlist Probability │    Compare & Filter   │    Weather & Platform   │
├─────────────────────────┼───────────────────────┼─────────────────────────┤
│ 4. Dynamic Fare Surge   │ 5. 24/7 AI Assistant  │ 6. Anti-Bot Fraud Shield│
│    Best Day to Book     │    Planner + Live PNR │    Tatkal Scalper Block │
└─────────────────────────┴───────────────────────┴─────────────────────────┘
```

### 1. 🎫 Ticket Confirmation Predictor
- **What it does**: Tells the passenger the exact likelihood that their waitlisted ticket will confirm:
  > *"Your WL 12 on 3A has an 82% chance of confirming before chart preparation."*
- **How it works**: Calibrated classification pipeline utilizing travel class, quota tier, booking horizon (days before departure), and route demand.
- **Explainability**: Integrated **TreeSHAP** explanations translating mathematical log-odds into transparent, plain-English reasons (e.g. *"General quota historically sees higher churn on this route"*).

### 2. 🔍 Train Recommender & Multi-Train Comparison
- **What it does**: Helps passengers choose between competing trains on busy corridors (e.g. Delhi–Mumbai, Delhi–Pune, Bangalore–Chennai).
- **Interactive Controls**:
  - Filter by priority: **Fastest**, **Cheapest**, **Highest Cleanliness**, or **Best Punctuality**.
  - Multi-train comparison drawer: Select up to 3 trains and compare departure times, duration, sleeper/AC fares, cleanliness ratings, and punctuality side-by-side.
- **Data Source**: Verified timetable database covering Rajdhani, Shatabdi, Duronto, Vande Bharat, and Superfast express services.

### 3. ⏱️ Train Delay & Station Congestion Predictor
- **What it does**: Predicts arrival delays in minutes with a statistical confidence range and station crowd levels:
  > *"Expected delay: ~25 minutes (likely between 15–35 minutes). Platform crowd: Moderate."*
- **Features Analyzed**: Route distance, scheduled departure hour, day of the week, rainfall intensity (mm), previous junction delay, and route congestion index.
- **Proactive Passenger Advisory**: Warns travelers if weather conditions (monsoon rains, winter fog) or heavy freight traffic require buffer time for connecting journeys.

### 4. 📈 Dynamic Fare Movement & Demand Surge Predictor
- **What it does**: Predicts ticket price trajectories and demand spikes across a 0–60 day booking window:
  > *"Current estimated fare is ₹1,420. Booking 14 days in advance will save you ~₹350."*
- **Features**:
  - 60-day visual price movement curve identifying the optimal day to book.
  - Quoted price anomaly verification to protect passengers from fraudulent third-party surge markups.
  - Festival & seasonal travel surge factor detection.

### 5. 💬 24/7 AI Travel Assistant & Unified Smart Planner
- **What it does**: An all-in-one conversational railway co-pilot supporting **English and हिंदी (देवनागरी)**:
  - 🚆 **Smart Trip Planning**: Natural language journey requests like *"Pune to Delhi tomorrow under ₹1500, reach before 10 AM"* automatically generate constraint-satisfying train cards with departure/arrival times, fares, savings, and alternative options.
  - 🔍 **Live RapidAPI PNR Tracking**: Enter any 10-digit PNR to retrieve real-time booking status, current coach/berth confirmation, and chart preparation status with verified offline fallbacks.
  - 📜 **Extractive Policy FAQ**: Zero-hallucination answers on cancellation penalties, Tatkal timing rules, luggage allowances, and RAC seat-sharing guidelines.
  - 🛡️ **Grounding & Privacy**: Automatic PII masking of 10-digit PNR numbers in audit logs (`******1029`) and autonomous human escalation for out-of-scope inquiries.

### 6. 🛡️ Anti-Cheat & Anti-Bot System (Tatkal Protection)
- **What it does**: Protects genuine passengers during peak Tatkal booking windows by detecting and blocking automated scalper bots.
- **Threat Detection**: Evaluates submission speed (e.g. form filled in < 2.5 seconds), device IP switching, automated keystroke velocity, and high-frequency booking attempts.
- **Explainability**: SHAP behavioral breakdown detailing whether risk was triggered by abnormal form-completion velocity, account volume, or bot-like session patterns.

---

## 🏛️ Enterprise AI Governance Architecture

YatraGov AI adheres to responsible AI governance principles, ensuring models remain fair, reliable, explainable, and compliant:

```
Telemetry Stream ──────> Covariate Drift Analysis (PSI & KS-Test)
                                │
                                ▼
                       Threshold Exceeded?
                         ├── YES ──> Autonomous Retraining & Canary Evaluation
                         └── NO  ──> Continuous Monitoring
                                │
                                ▼
                       Immutable Audit Trail (SQLite with Abort Triggers)
```

1. **Statistical Covariate Drift**: Real-time **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests run continuously across production and baseline data streams.
2. **Central Model Registry**: Tracks model artifacts, semantic versions (`v1`, `v2`), holdout metrics, and supports instant, zero-downtime rollback.
3. **Autonomous Supervisor Agent**: Periodically inspects production drift reports, triggers candidate retraining on unseen holdouts, compares against baseline performance, and safely deploys improved models.
4. **Immutable Audit Trail**: SQLite database guarded with SQL triggers that abort any `UPDATE` or `DELETE` statements, ensuring an untampered record of all inferences, feedback, and drift alerts.

---

## 💻 Technology Stack

- **Application & Dashboard**: [Streamlit](https://streamlit.io/) with custom Indian Railways Departure Board & Ticket styling
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/)
- **Explainable AI (XAI)**: [TreeSHAP](https://shap.readthedocs.io/)
- **Text & Multilingual Retrieval**: Character n-gram TF-IDF vectorization with Devanagari Unicode script detection
- **External Integration**: [RapidAPI](https://rapidapi.com/) for live Indian Railways PNR tracking
- **Testing**: [Pytest](https://pytest.org/) (63 automated tests covering models, APIs, drift, and UI components)

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- [Conda](https://docs.conda.io/en/latest/) or standard virtualenv

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/gunjan-07-dhiman/railgov-mvp.git
cd railgov-mvp

# Create and activate conda environment
conda create -n railgov python=3.11 -y
conda activate railgov

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Live RapidAPI)
Create a `.env` file in the root directory if you wish to use live RapidAPI PNR status:
```env
RAPIDAPI_KEY=your_rapidapi_key_here
RAPIDAPI_HOST=irctc-indian-railway-pnr-status.p.rapidapi.com
```
*(If omitted, the platform seamlessly uses verified demo databases without errors).*

### 3. Launch the Application
```bash
conda run -n railgov streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 4. Run the Test Suite
```bash
conda run -n railgov pytest tests/
```
Output: **`63 passed in ~13s`**

---

## 📁 Repository Structure

```
railgov-mvp/
├── app/
│   ├── base.py                   # ModelService abstract base class & registry
│   ├── config.py                 # Central configurations & MODULE_CATALOG
│   ├── modules/                  # The 6 core prediction & governance engines
│   │   ├── ticket_predictor.py   # Confirmation probability & TreeSHAP
│   │   ├── recommender.py        # Train recommender & comparison logic
│   │   ├── delay.py              # Delay regression & station congestion
│   │   ├── fare.py               # Fare prediction & 60-day horizon scans
│   │   ├── support_chat.py       # Multilingual chat, PNR & trip planner
│   │   ├── fraud.py              # Anti-bot detection & session velocity
│   │   └── planner.py            # Constraint-satisfaction itinerary planner
│   ├── governance/               # Drift detection (PSI/KS) & audit database
│   ├── services/                 # Live RapidAPI client with offline fallback
│   └── supervisor.py             # Autonomous monitoring & retraining loop
├── dashboard/
│   ├── app.py                    # Main departure board & feature navigation
│   ├── components/               # Railway theme, sidebar & reusable cards
│   └── pages/                    # Interactive web pages for all modules
├── data/                         # Verified timetable, baseline & drift data
├── docs/faq/                     # Bilingual IRCTC rulebook (EN & HI markdown)
├── tests/                        # Comprehensive unit & governance test suite
├── RUN.md                        # Quick-start instructions for reviewers
└── README.md                     # Project documentation
```

---

## ⚖️ License & Disclaimer

- **License**: Released under the [MIT License](LICENSE).
- **Disclaimer**: *YatraGov AI is an academic and research prototype developed for demonstration and educational purposes. It is not officially affiliated with or endorsed by Indian Railways, IRCTC, or CRIS.*