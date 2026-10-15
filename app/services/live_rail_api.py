"""
app/services/live_rail_api.py
Third-party live Indian Railways API client (RapidAPI integration).
Fetches real-time PNR status, seat allocations, and train schedules when credentials are provided in .env.
Falls back gracefully if offline, rate-limited, or when credentials are absent.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import requests

from app.config import BASE_DIR, RAPIDAPI_HOST, RAPIDAPI_KEY


def get_live_credentials() -> Dict[str, str]:
    """Retrieve API credentials, dynamically reading from .env if updated at runtime."""
    key = os.getenv("RAPIDAPI_KEY") or os.getenv("X_RAPIDAPI_KEY") or RAPIDAPI_KEY
    host = os.getenv("RAPIDAPI_HOST") or os.getenv("X_RAPIDAPI_HOST") or RAPIDAPI_HOST

    # Reload from .env file directly if not set in environment
    if not key:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k_clean = k.strip()
                    v_clean = v.strip().strip('"').strip("'")
                    if k_clean in ["RAPIDAPI_KEY", "X_RAPIDAPI_KEY"]:
                        key = v_clean
                    elif k_clean in ["RAPIDAPI_HOST", "X_RAPIDAPI_HOST"]:
                        host = v_clean

    return {"key": key.strip(), "host": host.strip()}


def fetch_live_pnr_status(pnr: str) -> Optional[Dict[str, Any]]:
    """
    Fetch real-time PNR status from RapidAPI.
    Supports popular Indian Railways endpoints with dynamic schema normalization.
    """
    creds = get_live_credentials()
    api_key = creds["key"]
    api_host = creds["host"]

    if not api_key:
        return None

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host,
    }

    # Determine candidate URLs based on the host configuration
    urls_to_try = [
        f"https://{api_host}/getPNRStatus/{pnr}",
        f"https://{api_host}/api/v3/getPNRStatus?pnrNumber={pnr}",
        f"https://{api_host}/pnr-status/{pnr}",
        f"https://{api_host}/api/pnr-status?pnr={pnr}",
    ]

    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=headers, timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("success") is False and "Flushed" in data.get("message", ""):
                    return {
                        "pnr_number": pnr,
                        "passenger_name": "N/A",
                        "train_number": "N/A",
                        "train_name": "Indian Railways",
                        "travel_class": "N/A",
                        "journey_date": "Expired / Flushed",
                        "booking_status": "Flushed / Not Generated",
                        "current_status": "Flushed / Not Generated",
                        "chart_status": "Flushed PNR or Not Yet Generated in Indian Railways PRS",
                        "refund_status": "N/A",
                        "is_live": True,
                    }
                parsed = _normalize_pnr_payload(pnr, data)
                if parsed:
                    return parsed
        except Exception:
            continue

    return None


def _normalize_pnr_payload(pnr: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize vendor-specific RapidAPI response schemas into standardized dictionary."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload

    train_num = str(data.get("TrainNo") or data.get("train_number") or data.get("train_no") or "Live Express")
    train_name = str(data.get("TrainName") or data.get("train_name") or "Express")
    doj = str(data.get("Doj") or data.get("date_of_journey") or data.get("doj") or "Upcoming")
    travel_class = str(data.get("Class") or data.get("journey_class") or data.get("class") or "SL")
    chart_status = str(data.get("ChartPrepared") or data.get("chart_status") or "Chart Not Prepared")

    passengers = data.get("PassengerStatus") or data.get("passengers") or []
    current_status = "CNF"
    booking_status = "CNF"

    if isinstance(passengers, list) and len(passengers) > 0:
        p1 = passengers[0]
        if isinstance(p1, dict):
            current_status = p1.get("CurrentStatus") or p1.get("current_status") or "CNF"
            booking_status = p1.get("BookingStatus") or p1.get("booking_status") or current_status
    elif "CurrentStatus" in data:
        current_status = str(data["CurrentStatus"])

    return {
        "pnr_number": pnr,
        "passenger_name": "Verified Passenger",
        "train_number": train_num,
        "train_name": train_name,
        "travel_class": travel_class,
        "journey_date": doj,
        "booking_status": booking_status,
        "current_status": current_status,
        "chart_status": chart_status,
        "refund_status": "N/A",
        "is_live": True,
    }
