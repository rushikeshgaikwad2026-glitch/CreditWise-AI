"""Schema-driven validation and prediction helpers for batch assessment."""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data_loader import PROJECT_ROOT
from src.model_service import predict_default_probability, validate_applicant
from src.validation import validate_form_values

OPTIONAL_MODEL_FIELDS = {"person_emp_length", "loan_int_rate"}
DERIVED_MODEL_FIELDS = {"loan_percent_income"}
INTEGER_MODEL_FIELDS = {"person_age", "cb_person_cred_hist_length"}
NUMERIC_MODEL_FIELDS = {
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
}
REFERENCE_REPORT_PATH = PROJECT_ROOT / "reports" / "phase4_optimization_calibration_report.json"


def required_upload_columns(model_features: list[str]) -> list[str]:
    """Return required user-supplied columns, excluding imputed and derived inputs."""
    optional_fields = OPTIONAL_MODEL_FIELDS | DERIVED_MODEL_FIELDS
    return [feature for feature in model_features if feature not in optional_fields]


def build_sample_template(model_features: list[str]) -> pd.DataFrame:
    """Create a header-only CSV template from the saved model's actual input schema."""
    return pd.DataFrame(columns=model_features)


def inspect_upload_columns(
    uploaded_columns: list[str], model_features: list[str]
) -> dict[str, list[str]]:
    """Compare upload columns with model features and supported optional fields."""
    required = required_upload_columns(model_features)
    return {
        "missing_required": [column for column in required if column not in uploaded_columns],
        "missing_optional": [
            column
            for column in model_features
            if column in OPTIONAL_MODEL_FIELDS | DERIVED_MODEL_FIELDS
            and column not in uploaded_columns
        ],
        "unexpected": [column for column in uploaded_columns if column not in model_features],
    }


@st.cache_data(show_spinner=False)
def load_reference_threshold(
    path: str = str(REFERENCE_REPORT_PATH),
) -> tuple[float | None, str | None]:
    """Load the previously selected training-OOF F1 threshold for transparent labels."""
    report_path = Path(path)
    if not report_path.is_file():
        return None, "The saved model report is unavailable; using illustrative probability bands."
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        threshold = float(report["candidate_threshold_selected_on_training_oof_f1"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The saved threshold must be between zero and one.")
        rule = str(report.get("threshold_selection_rule", "selected on training-only out-of-fold data"))
        return threshold, rule
    except Exception as error:
        return None, f"The saved threshold could not be read ({error}); using illustrative probability bands."


def categorize_probability(probability: float, reference_threshold: float | None) -> str:
    """Assign a transparent reporting band without making a lending decision."""
    if reference_threshold is not None:
        threshold_label = f"{reference_threshold:.0%} training-OOF reference"
        if probability < reference_threshold:
            return f"Below {threshold_label} threshold"
        return f"At or above {threshold_label} threshold"

    if probability < 0.10:
        return "Lower probability (illustrative <10%)"
    if probability < 0.25:
        return "Moderate probability (illustrative 10-25%)"
    if probability < 0.50:
        return "Elevated probability (illustrative 25-50%)"
    return "Higher probability (illustrative 50%+)"


def risk_category_labels(reference_threshold: float | None) -> list[str]:
    """Return all display bands in order, including bands with zero records."""
    if reference_threshold is not None:
        threshold_label = f"{reference_threshold:.0%} training-OOF reference"
        return [
            f"Below {threshold_label} threshold",
            f"At or above {threshold_label} threshold",
        ]
    return [
        "Lower probability (illustrative <10%)",
        "Moderate probability (illustrative 10-25%)",
        "Elevated probability (illustrative 25-50%)",
        "Higher probability (illustrative 50%+)",
    ]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _numeric_value(
    value: Any,
    *,
    field: str,
    allow_missing: bool,
) -> tuple[float | None, str | None]:
    if _is_missing(value):
        if allow_missing:
            return float("nan"), None
        return None, "A value is required."
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, "Must be numeric."
    if not math.isfinite(parsed):
        return None, "Must be a finite number."
    if field in INTEGER_MODEL_FIELDS and not parsed.is_integer():
        return None, "Must be a whole number."
    return parsed, None


def _validation_field(message: str) -> str:
    field_map = (
        ("employment length", "person_emp_length"),
        ("credit history length", "cb_person_cred_hist_length"),
        ("annual income", "person_income"),
        ("loan amount", "loan_amnt"),
        ("interest rate", "loan_int_rate"),
        ("loan_percent_income", "loan_percent_income"),
        ("age", "person_age"),
    )
    lower_message = message.lower()
    return next((field for phrase, field in field_map if phrase in lower_message), "Applicant record")


def validate_batch_records(
    uploaded: pd.DataFrame,
    model_features: list[str],
    categories: dict[str, list[str]],
) -> dict[str, Any]:
    """Validate records independently and produce model-ordered rows for valid applicants."""
    columns = inspect_upload_columns(list(uploaded.columns), model_features)
    errors: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []

    for position in range(len(uploaded)):
        source_row = uploaded.iloc[position]
        original_values = source_row.to_dict()
        row_number = position + 2
        row_errors: list[dict[str, Any]] = []

        for field in columns["missing_required"]:
            row_errors.append(
                {
                    "Original row number": row_number,
                    "Invalid field": field,
                    "Reason": "Required model input column is missing from the uploaded CSV.",
                    "Uploaded value": None,
                }
            )
        if row_errors:
            errors.extend(row_errors)
            continue

        prepared_values: dict[str, Any] = {}
        parse_failed = False
        for feature in model_features:
            raw_value = source_row.get(feature)
            if feature in NUMERIC_MODEL_FIELDS and feature not in DERIVED_MODEL_FIELDS:
                parsed, error = _numeric_value(
                    raw_value,
                    field=feature,
                    allow_missing=feature in OPTIONAL_MODEL_FIELDS,
                )
                if error:
                    row_errors.append(
                        {
                            "Original row number": row_number,
                            "Invalid field": feature,
                            "Reason": error,
                            "Uploaded value": raw_value,
                        }
                    )
                    parse_failed = True
                prepared_values[feature] = parsed
            elif feature in DERIVED_MODEL_FIELDS:
                prepared_values[feature] = None
            else:
                if _is_missing(raw_value):
                    row_errors.append(
                        {
                            "Original row number": row_number,
                            "Invalid field": feature,
                            "Reason": "A categorical value is required.",
                            "Uploaded value": raw_value,
                        }
                    )
                    parse_failed = True
                    continue
                value = str(raw_value).strip()
                if value not in categories.get(feature, []):
                    allowed = ", ".join(categories.get(feature, []))
                    row_errors.append(
                        {
                            "Original row number": row_number,
                            "Invalid field": feature,
                            "Reason": f"Unsupported category. Expected one of: {allowed}.",
                            "Uploaded value": raw_value,
                        }
                    )
                    parse_failed = True
                prepared_values[feature] = value

        if parse_failed:
            errors.extend(row_errors)
            continue

        annual_income = prepared_values.get("person_income")
        loan_amount = prepared_values.get("loan_amnt")
        form_errors = validate_form_values(
            age=int(prepared_values["person_age"]),
            annual_income=float(annual_income),
            employment_length=(
                None
                if pd.isna(prepared_values.get("person_emp_length", float("nan")))
                else float(prepared_values["person_emp_length"])
            ),
            loan_amount=float(loan_amount),
            interest_rate=(
                None
                if pd.isna(prepared_values.get("loan_int_rate", float("nan")))
                else float(prepared_values["loan_int_rate"])
            ),
            credit_history_length=int(prepared_values["cb_person_cred_hist_length"]),
        )
        for message in form_errors:
            field = _validation_field(message)
            row_errors.append(
                {
                    "Original row number": row_number,
                    "Invalid field": field,
                    "Reason": message,
                    "Uploaded value": original_values.get(field),
                }
            )

        if annual_income is not None and loan_amount is not None and annual_income > 0:
            derived_ratio = loan_amount / annual_income
            if not math.isfinite(derived_ratio) or derived_ratio <= 0:
                row_errors.append(
                    {
                        "Original row number": row_number,
                        "Invalid field": "loan_percent_income",
                        "Reason": "The derived loan-to-income ratio must be finite and greater than zero.",
                        "Uploaded value": original_values.get("loan_percent_income"),
                    }
                )
            else:
                supplied_ratio = original_values.get("loan_percent_income")
                if not _is_missing(supplied_ratio):
                    parsed_ratio, ratio_error = _numeric_value(
                        supplied_ratio,
                        field="loan_percent_income",
                        allow_missing=False,
                    )
                    if ratio_error:
                        row_errors.append(
                            {
                                "Original row number": row_number,
                                "Invalid field": "loan_percent_income",
                                "Reason": ratio_error,
                                "Uploaded value": supplied_ratio,
                            }
                        )
                    elif parsed_ratio <= 0 or abs(parsed_ratio - derived_ratio) > 0.0051:
                        row_errors.append(
                            {
                                "Original row number": row_number,
                                "Invalid field": "loan_percent_income",
                                "Reason": (
                                    "Must match loan amount divided by annual income; values rounded "
                                    "to two decimal places are accepted."
                                ),
                                "Uploaded value": supplied_ratio,
                            }
                        )
                prepared_values["loan_percent_income"] = derived_ratio

        if row_errors:
            errors.extend(row_errors)
            continue

        try:
            model_input = validate_applicant(prepared_values, model_features, categories)
        except (TypeError, ValueError) as error:
            row_errors.append(
                {
                    "Original row number": row_number,
                    "Invalid field": _validation_field(str(error)),
                    "Reason": str(error),
                    "Uploaded value": original_values.get(_validation_field(str(error))),
                }
            )
            errors.extend(row_errors)
            continue

        warnings = []
        if prepared_values["loan_percent_income"] > 1:
            warnings.append(
                "Loan amount exceeds annual income; as in single assessment, this is an advisory, not an automatic rejection."
            )
        valid_records.append(
            {
                "Original row number": row_number,
                "Original input": original_values,
                "Model input": model_input,
                "Validation warnings": warnings,
            }
        )

    return {
        "total_records": len(uploaded),
        "missing_required_columns": columns["missing_required"],
        "missing_optional_columns": columns["missing_optional"],
        "unexpected_columns": columns["unexpected"],
        "valid_records": valid_records,
        "validation_errors": pd.DataFrame(
            errors,
            columns=["Original row number", "Invalid field", "Reason", "Uploaded value"],
        ),
    }


def run_batch_predictions(
    uploaded: pd.DataFrame,
    model: Any,
    model_features: list[str],
    categories: dict[str, list[str]],
    reference_threshold: float | None,
) -> dict[str, Any]:
    """Validate records, assess valid applicants, and build downloadable result rows."""
    batch = validate_batch_records(uploaded, model_features, categories)
    valid_records = []
    prediction_errors: list[dict[str, Any]] = []

    for record in batch["valid_records"]:
        try:
            probability = predict_default_probability(model, record["Model input"])
        except Exception as error:
            prediction_errors.append(
                {
                    "Original row number": record["Original row number"],
                    "Invalid field": "Prediction",
                    "Reason": f"The saved model could not assess this record: {error}",
                    "Uploaded value": None,
                }
            )
            continue

        category = categorize_probability(probability, reference_threshold)
        valid_records.append(
            {
                **record,
                "Estimated default probability": probability,
                "Risk category": category,
                "Assessment status": "Assessed",
            }
        )

    errors = pd.concat(
        [batch["validation_errors"], pd.DataFrame(prediction_errors)],
        ignore_index=True,
    )
    valid_by_row = {record["Original row number"]: record for record in valid_records}
    result_rows = []
    for position in range(len(uploaded)):
        row_number = position + 2
        original_values = uploaded.iloc[position].to_dict()
        record = valid_by_row.get(row_number)
        preserved_values = {
            key if key != "Original row number" else "Uploaded input: Original row number": value
            for key, value in original_values.items()
        }
        if record is None:
            matching_errors = errors.loc[errors["Original row number"] == row_number]
            status = "Invalid: " + "; ".join(matching_errors["Reason"].astype(str).tolist())
            result = {
                "Original row number": row_number,
                **preserved_values,
                "Estimated default probability": None,
                "Risk category": "Not assessed",
                "Assessment status": status,
            }
        else:
            result = {
                "Original row number": row_number,
                **preserved_values,
                "Estimated default probability": record["Estimated default probability"],
                "Risk category": record["Risk category"],
                "Assessment status": record["Assessment status"],
            }
        result_rows.append(result)

    batch.update(
        {
            "valid_records": valid_records,
            "invalid_count": len(uploaded) - len(valid_records),
            "valid_count": len(valid_records),
            "results": pd.DataFrame(result_rows),
            "validation_errors": errors,
        }
    )
    return batch