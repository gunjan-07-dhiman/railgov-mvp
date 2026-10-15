# Quick Start Guide: YatraGov AI

Follow these simple steps to run and test the application in your browser.

---

## 1. Open the Web App

In your terminal, run this single command:

```bash
conda run -n railgov streamlit run dashboard/app.py
```

Then open your browser to: **[http://localhost:8501](http://localhost:8501)**

---

## 2. How to Test Each Feature in the UI

Once the website opens, use the **dark navy sidebar on the left** to click any page:

### 💬 Support Chatbot (With Live RapidAPI)
- **Click**: `Support Chatbot` in the sidebar.
- **How to test**:
  - **Live PNR**: Type `Check PNR <your-10-digit-pnr>` in the bottom chat box and press Enter. It connects directly to RapidAPI and displays your real coach, berth, status, and charting status.
  - **Demo PNR**: Type `Check PNR 9823471029`.
  - **Hindi Query**: Click the chip *"तत्काल टिकट बुकिंग का समय क्या है?"*.
  - **Out-of-Scope Test**: Ask *"Can I order pizza?"* to see the bot safely decline and offer human escalation.
  - **Feedback**: Click *"Helpful"* or *"Not helpful"* under any reply.

### 🎫 Ticket Confirmation Predictor
- **Click**: `Ticket Confirmation` in the sidebar.
- **How to test**: Pick your class (e.g. `3A`), select `Waitlist (WL)`, set queue position `18`, and click **Predict Confirmation Odds**.
- **What you see**: A speedometer-style confirmation gauge (e.g. *82% Likely Confirmed*) and plain-English reasons explaining why.

### ⏱️ Train Delay & Platform Crowd
- **Click**: `Train Delay & Platform` in the sidebar.
- **How to test**: Choose train type (`Superfast`), enter distance `850 km`, and click **Predict Delay & Crowd**.
- **What you see**: Expected delay in minutes (e.g. *~25 min, likely 15–35 min*) and platform crowd status.

### 📈 Fare & Demand Surge
- **Click**: `Fare & Demand Surge` in the sidebar.
- **How to test**: Select class (`3A`), distance (`1200 km`), lead days (`5 days`), and click **Analyze Fare Movement**.
- **What you see**: Estimated fare, price anomaly check, and a 60-day price graph showing the best day to book and how much you save.

### 🛡️ Booking Fraud Shield
- **Click**: `Fraud Shield` in the sidebar.
- **How to test**: Set form fill to `1.2s`, bookings to `10`, accounts on device to `6`, and click **Evaluate Session Integrity**.
- **What you see**: Threat classification immediately turns **HIGH RISK** (red light) and logs an alert in the analyst table below.

### ⚙️ AI Governance Console
- **Click**: `Governance Console` at the bottom of the sidebar.
- **How to test**: Use the top dropdown to inspect any model (`support_chatbot`, `fraud_detector`, `fare_predictor`, etc.) across 5 tabs: Health, Drift, Registry, Audit Trail, and Autonomous Supervisor.

---

## 3. Run the Automated Tests

To verify that all 63 tests are passing:

```bash
conda run -n railgov pytest tests/
```
*(All 63 tests should pass with 100% success)*

