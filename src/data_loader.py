"""Cached access to the cleaned portfolio dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "credit_risk_cleaned.csv"

REQUIRED_PORTFOLIO_COLUMNS = {
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
}


@st.cache_data(show_spinner="Loading the cleaned credit portfolio...")
def load_portfolio_data(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load the cleaned dataset and validate its expected dashboard fields."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Cleaned portfolio dataset was not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing_columns = REQUIRED_PORTFOLIO_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "The cleaned portfolio dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if frame.empty:
        raise ValueError("The cleaned portfolio dataset contains no records.")
    if not frame["loan_status"].isin([0, 1]).all():
        raise ValueError("loan_status must contain only 0 (non-default) and 1 (default).")
    return frame
