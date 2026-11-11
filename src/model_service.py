"""Loading, schema discovery, and prediction for the saved candidate model."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_risk_rf_optimized_calibrated_no_loan_grade.joblib"
)


@st.cache_resource(show_spinner="Loading the calibrated risk model...")
def load_candidate_model(path: str = str(MODEL_PATH)) -> Any:
    """Load the preserved model artifact and check its prediction contract."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Candidate model was not found: {model_path}")

    model = joblib.load(model_path)
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("The candidate model does not expose predict_proba().")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("The saved candidate model does not expose its input feature names.")

    feature_names = list(model.feature_names_in_)
    if "loan_grade" in feature_names:
        raise ValueError("The provisional candidate must not use loan_grade as an input.")
    if 1 not in model.classes_:
        raise ValueError("The candidate model does not contain the default class (1).")
    return model


def get_model_schema(model: Any) -> tuple[list[str], dict[str, list[str]]]:
    """Return original model features and valid categories from fitted preprocessing."""
    feature_names = list(model.feature_names_in_)
    calibrated_folds = getattr(model, "calibrated_classifiers_", None)
    if not calibrated_folds:
        raise ValueError("The candidate model has no fitted calibrated estimators.")

    estimator = calibrated_folds[0].estimator
    if not isinstance(estimator, Pipeline) or "preprocessor" not in estimator.named_steps:
        raise ValueError("The saved model pipeline does not contain its fitted preprocessor.")

    preprocessor = estimator.named_steps["preprocessor"]
    categorical_transformer = preprocessor.named_transformers_.get("categorical")
    if categorical_transformer is None:
        raise ValueError("The fitted preprocessor has no categorical transformer.")

    encoder = categorical_transformer.named_steps.get("onehot")
    categorical_columns = next(
        columns
        for name, _, columns in preprocessor.transformers_
        if name == "categorical"
    )
    if encoder is None or not hasattr(encoder, "categories_"):
        raise ValueError("The fitted categorical transformer has no learned categories.")

    categories = {
        column: [str(value) for value in values]
        for column, values in zip(categorical_columns, encoder.categories_, strict=True)
    }
    return feature_names, categories


def validate_applicant(
    applicant: dict[str, Any],
    feature_names: list[str],
    categorical_values: dict[str, list[str]],
) -> pd.DataFrame:
    """Validate a single applicant and produce a one-row model input frame."""
    missing = [feature for feature in feature_names if feature not in applicant]
    extra = [feature for feature in applicant if feature not in feature_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing required fields: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected fields: {', '.join(extra)}")
        raise ValueError("Applicant input does not match the model schema (" + "; ".join(details) + ").")

    row: dict[str, Any] = {}
    for feature in feature_names:
        value = applicant[feature]
        if feature in categorical_values:
            if value not in categorical_values[feature]:
                raise ValueError(
                    f"{feature} must be one of: {', '.join(categorical_values[feature])}."
                )
            row[feature] = value
            continue

        if value is None or pd.isna(value):
            if feature not in {"person_emp_length", "loan_int_rate"}:
                raise ValueError(f"{feature} is required and cannot be blank.")
            row[feature] = np.nan
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must be numeric.") from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"{feature} must be a finite number.")
        row[feature] = numeric_value

    for feature in ("person_age", "person_income", "loan_amnt"):
        if feature in row and row[feature] <= 0:
            raise ValueError(f"{feature} must be greater than zero.")
    for feature in ("person_emp_length", "loan_int_rate", "loan_percent_income"):
        if feature in row and pd.notna(row[feature]) and row[feature] < 0:
            raise ValueError(f"{feature} cannot be negative.")
    if row.get("loan_percent_income", 0) <= 0:
        raise ValueError("loan_percent_income must be greater than zero.")
    if row.get("person_emp_length") is not None and pd.notna(row.get("person_emp_length")):
        if row["person_emp_length"] > row["person_age"]:
            raise ValueError("Employment length cannot exceed the applicant's age.")
    if row.get("cb_person_cred_hist_length", 0) > row.get("person_age", 0):
        raise ValueError("Credit history length cannot exceed the applicant's age.")

    return pd.DataFrame([row], columns=feature_names)


def predict_default_probability(model: Any, model_input: pd.DataFrame) -> float:
    """Return the saved calibrated model's probability for class 1."""
    probabilities = model.predict_proba(model_input)
    class_index = list(model.classes_).index(1)
    probability = float(probabilities[0, class_index])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The candidate model returned an invalid default probability.")
    return probability
