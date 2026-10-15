"""
data/generate_chat_data.py
Synthetic interaction telemetry generator for Support Chatbot governance monitoring.

# Note: All interaction telemetry generated here represents simulated traffic for academic governance benchmarking.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from app.config import (
    CHAT_BASELINE_PATH,
    CHAT_DRIFTED_PATH,
    DATA_DIR,
)


def generate_chat_telemetry(
    n_samples: int = 300,
    is_drifted: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic chatbot interaction telemetry batches for governance monitoring."""
    rng = np.random.RandomState(random_seed)

    in_scope_queries_en = [
        "What is the refund rule on ticket cancellation?",
        "When does Tatkal booking open for AC coaches?",
        "What does RAC status mean on my train ticket?",
        "Can a passenger travel on a waitlisted e-ticket?",
        "What are the clerkage charges for ticket cancellation?",
        "How do I check PNR status on this platform?",
        "When is the reservation chart prepared before departure?",
        "What happens if my train is cancelled by the railway?",
    ]
    in_scope_queries_hi = [
        "कन्फर्म टिकट कैंसिल करने पर कितना रिफंड मिलता है?",
        "तत्काल टिकट बुकिंग किस समय शुरू होती है?",
        "पीएनआर में सीएनएफ और आरएसी का क्या मतलब है?",
        "क्या वेटलिस्ट ई-टिकट पर ट्रेन में यात्रा कर सकते हैं?",
        "वेटलिस्ट टिकट रद्द करने पर क्लर्केज चार्ज कितना लगता है?",
        "रेलवे द्वारा ट्रेन रद्द होने पर क्या पूरा रिफंड मिलता है?",
        "आरएसी और वेटिंग टिकट में क्या अंतर होता है?",
        "चार्ट बनने के बाद टिकट का स्टेटस कैसे चेक करें?",
    ]
    out_of_scope_queries_en = [
        "Can I order hot pizza delivered to my seat at the station?",
        "How do I book a domestic flight from Delhi to Mumbai?",
        "What is the current cricket match score?",
        "Tell me a chocolate cake recipe with eggless ingredients.",
        "What are the trending stock market picks for today?",
        "How do I apply for an international tourist visa?",
    ]
    out_of_scope_queries_hi = [
        "क्या मुझे स्टेशन पर होटल का कमरा बुक करना है?",
        "पासपोर्ट रिन्यूअल के लिए आवश्यक दस्तावेज क्या हैं?",
        "पनीर टिक्का बनाने की आसान विधि क्या है?",
        "ऑनलाइन ड्राइविंग लाइसेंस का फॉर्म कैसे भरें?",
        "आज के शेयर बाजार का मुख्य समाचार बताएं।",
    ]

    records = []
    for i in range(n_samples):
        if not is_drifted:
            # Baseline: 85% in-scope travel questions, 15% out-of-scope
            is_out = rng.rand() < 0.15
            lang = "hi" if rng.rand() < 0.35 else "en"
            has_pnr = 1 if rng.rand() < 0.18 else 0
        else:
            # Drifted: 60% out-of-scope queries (topic drift / user confusion), 40% in-scope
            is_out = rng.rand() < 0.60
            lang = "hi" if rng.rand() < 0.40 else "en"
            has_pnr = 1 if rng.rand() < 0.08 else 0

        if has_pnr:
            fake_pnr = rng.randint(1000000000, 9999999999)
            if lang == "en":
                q = f"Please check my PNR status {fake_pnr}"
            else:
                q = f"कृपया मेरे पीएनआर नंबर {fake_pnr} की स्थिति बताएं"
            sim = rng.uniform(0.70, 0.95)
            grounded = 1
            escalated = 0
        elif not is_out:
            q = rng.choice(in_scope_queries_hi if lang == "hi" else in_scope_queries_en)
            sim = rng.uniform(0.32, 0.68)
            grounded = 1
            escalated = 0
        else:
            q = rng.choice(out_of_scope_queries_hi if lang == "hi" else out_of_scope_queries_en)
            sim = rng.uniform(0.05, 0.22)
            grounded = 0
            escalated = 1

        records.append({
            "query_id": f"q_{i+1:04d}",
            "query_text": q,
            "language": lang,
            "char_length": len(q),
            "has_pnr": has_pnr,
            "similarity_score": round(float(sim), 4),
            "is_grounded": grounded,
            "is_escalated": escalated,
        })

    df = pd.DataFrame(records)
    return df


def generate_and_save_all(
    baseline_path: Path = CHAT_BASELINE_PATH,
    drifted_path: Path = CHAT_DRIFTED_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and write baseline and drifted chat telemetry to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_df = generate_chat_telemetry(n_samples=300, is_drifted=False, random_seed=42)
    drift_df = generate_chat_telemetry(n_samples=300, is_drifted=True, random_seed=101)

    base_df.to_csv(baseline_path, index=False)
    drift_df.to_csv(drifted_path, index=False)
    return base_df, drift_df


if __name__ == "__main__":
    b, d = generate_and_save_all()
    print(f"Chat baseline: {b.shape}, grounded rate: {b['is_grounded'].mean():.2%}")
    print(f"Chat drifted:  {d.shape}, grounded rate: {d['is_grounded'].mean():.2%}")
