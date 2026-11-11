"""Validated access to the saved Phase 4 SHAP explanations and plots."""

import io
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

GLOBAL_IMPORTANCE_PATH = REPORTS_DIR / "phase4_shap_global_importance.csv"
APPLICANT_EXPLANATIONS_CSV_PATH = REPORTS_DIR / "phase4_shap_applicant_explanations.csv"
APPLICANT_EXPLANATIONS_JSON_PATH = REPORTS_DIR / "phase4_shap_applicant_explanations.json"

GLOBAL_IMAGE_PATHS = {
    "global_importance": REPORTS_DIR / "phase4_shap_global_bar.png",
    "summary_beeswarm": REPORTS_DIR / "phase4_shap_summary_beeswarm.png",
}

PROFILE_IMAGE_PATHS = {
    "Lower estimated risk": REPORTS_DIR / "phase4_shap_lower_estimated_risk_waterfall.png",
    "Medium estimated risk": REPORTS_DIR / "phase4_shap_medium_estimated_risk_waterfall.png",
    "Higher estimated risk": REPORTS_DIR / "phase4_shap_higher_estimated_risk_waterfall.png",
}

PROFILE_ORDER = list(PROFILE_IMAGE_PATHS)


def validate_png_artifact(path: str | Path) -> tuple[bytes | None, str | None]:
    """Return verified PNG bytes or a user-safe error for a missing/corrupt image."""
    image_path = Path(path)
    if not image_path.is_file():
        return None, f"SHAP image is missing: {image_path.name}."

    try:
        image_bytes = image_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "PNG":
                return None, f"SHAP image is not a PNG file: {image_path.name}."
            image.verify()
    except Exception as error:
        return None, f"Could not read SHAP image {image_path.name}: {error}"
    return image_bytes, None


@st.cache_data(show_spinner=False)
def load_png_artifact(path: str) -> tuple[bytes | None, str | None]:
    """Cache a verified PNG artifact by path."""
    return validate_png_artifact(path)


@st.cache_data(show_spinner=False)
def load_global_importance(
    path: str = str(GLOBAL_IMPORTANCE_PATH),
) -> tuple[pd.DataFrame | None, str | None]:
    """Load, validate, and rank the saved mean-absolute-SHAP table."""
    importance_path = Path(path)
    if not importance_path.is_file():
        return None, f"Global SHAP importance data is missing: {importance_path.name}."

    try:
        importance = pd.read_csv(importance_path)
        required_columns = {"feature", "mean_absolute_shap_value"}
        missing_columns = required_columns.difference(importance.columns)
        if missing_columns:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing_columns)))
        if importance.empty:
            raise ValueError("The importance table contains no features.")
        importance = importance.loc[:, ["feature", "mean_absolute_shap_value"]].copy()
        importance["feature"] = importance["feature"].astype(str)
        importance["mean_absolute_shap_value"] = pd.to_numeric(
            importance["mean_absolute_shap_value"], errors="raise"
        )
        values = importance["mean_absolute_shap_value"].tolist()
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("SHAP importance values must be finite and non-negative.")
        if importance["feature"].duplicated().any():
            raise ValueError("The importance table contains duplicate feature names.")
        return (
            importance.sort_values(
                "mean_absolute_shap_value", ascending=False, kind="stable"
            ).reset_index(drop=True),
            None,
        )
    except Exception as error:
        return None, f"Could not read global SHAP importance data: {error}"


def _contributions_match(csv_value: str, json_items: list[dict[str, Any]]) -> bool:
    try:
        csv_items = json.loads(csv_value)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(csv_items, list) or len(csv_items) != len(json_items):
        return False

    for csv_item, json_item in zip(csv_items, json_items, strict=True):
        for key in ("feature", "display_name", "observed_value"):
            if csv_item.get(key) != json_item.get(key):
                return False
        try:
            if not math.isclose(
                float(csv_item["shap_contribution"]),
                float(json_item["shap_contribution"]),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _profile_value_matches(csv_value: Any, json_value: Any) -> bool:
    if json_value is None:
        return pd.isna(csv_value)
    if isinstance(json_value, (int, float)) and not isinstance(json_value, bool):
        try:
            return math.isclose(
                float(csv_value), float(json_value), rel_tol=0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return str(csv_value) == str(json_value)


@st.cache_data(show_spinner=False)
def load_applicant_explanations(
    csv_path: str = str(APPLICANT_EXPLANATIONS_CSV_PATH),
    json_path: str = str(APPLICANT_EXPLANATIONS_JSON_PATH),
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Load the JSON profile records and verify them against their CSV artifact."""
    csv_file, json_file = Path(csv_path), Path(json_path)
    if not csv_file.is_file():
        return None, f"Applicant SHAP CSV is missing: {csv_file.name}."
    if not json_file.is_file():
        return None, f"Applicant SHAP JSON is missing: {json_file.name}."

    try:
        table = pd.read_csv(csv_file)
        profiles = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(profiles, list) or not profiles:
            raise ValueError("The applicant JSON must contain a non-empty profile list.")
        required_csv_columns = {
            "profile",
            "top_positive_contributions",
            "top_negative_contributions",
        }
        missing_columns = required_csv_columns.difference(table.columns)
        if missing_columns:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing_columns)))
        if table["profile"].astype(str).duplicated().any():
            raise ValueError("The applicant CSV contains duplicate profile names.")
        if len(table) != len(profiles):
            raise ValueError("The CSV and JSON contain different numbers of profiles.")

        csv_profiles = {str(row["profile"]): row for row in table.to_dict(orient="records")}
        validated_profiles: list[dict[str, Any]] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                raise ValueError("Each applicant profile must be a JSON object.")
            name = profile.get("profile")
            if name not in PROFILE_ORDER or name not in csv_profiles:
                raise ValueError(f"Unexpected or unmatched applicant profile: {name!r}.")
            if name in [item["profile"] for item in validated_profiles]:
                raise ValueError(f"Duplicate applicant profile in JSON: {name}.")

            positive = profile.get("top_positive_contributions")
            negative = profile.get("top_negative_contributions")
            observed = profile.get("observed_features")
            if not isinstance(positive, list) or not isinstance(negative, list):
                raise ValueError(f"Contribution lists are missing for {name}.")
            if not isinstance(observed, dict):
                raise ValueError(f"Observed profile values are missing for {name}.")
            if any(not isinstance(item, dict) for item in positive + negative):
                raise ValueError(f"A contribution entry is invalid for {name}.")
            for item in positive + negative:
                contribution = float(item["shap_contribution"])
                if not math.isfinite(contribution):
                    raise ValueError(f"A non-finite SHAP value was found for {name}.")
            if any(float(item["shap_contribution"]) <= 0 for item in positive):
                raise ValueError(f"Positive contributions have invalid signs for {name}.")
            if any(float(item["shap_contribution"]) >= 0 for item in negative):
                raise ValueError(f"Negative contributions have invalid signs for {name}.")

            row = csv_profiles[name]
            for json_column, csv_column in (
                ("calibrated_candidate_default_probability", "calibrated_candidate_default_probability"),
                ("fold_1_underlying_rf_raw_default_probability", "underlying_fold_rf_raw_default_probability"),
                ("fold_1_rf_base_value", "fold_rf_base_value"),
            ):
                json_value = profile.get(json_column)
                if csv_column in table.columns and not _profile_value_matches(
                    row[csv_column], json_value
                ):
                    raise ValueError(f"CSV/JSON probability values do not match for {name}.")

            for contribution_key in (
                "top_positive_contributions",
                "top_negative_contributions",
            ):
                if not _contributions_match(row[contribution_key], profile[contribution_key]):
                    raise ValueError(f"CSV/JSON {contribution_key} do not match for {name}.")

            for feature, value in observed.items():
                if feature not in table.columns or not _profile_value_matches(row[feature], value):
                    raise ValueError(f"CSV/JSON observed value for {feature} does not match in {name}.")
            validated_profiles.append(profile)

        order = {name: index for index, name in enumerate(PROFILE_ORDER)}
        validated_profiles.sort(key=lambda item: order[item["profile"]])
        return validated_profiles, None
    except Exception as error:
        return None, f"Could not validate applicant SHAP explanations: {error}"


def display_feature_name(feature: str) -> str:
    """Format the saved encoded feature name for a business-facing table."""
    readable = feature.removeprefix("numeric__").removeprefix("categorical__")
    return readable.replace("_", " ").title()


def global_importance_observations(importance: pd.DataFrame) -> list[str]:
    """Generate concise observations from the saved global importance values."""
    if importance.empty:
        return []

    top = importance.iloc[0]
    observations = [
        f"{display_feature_name(str(top['feature']))} has the largest mean absolute SHAP value "
        f"({float(top['mean_absolute_shap_value']):.4f}); this indicates attribution magnitude, "
        "not whether the feature raises or lowers risk."
    ]
    total_importance = float(importance["mean_absolute_shap_value"].sum())
    top_three_importance = float(
        importance.head(3)["mean_absolute_shap_value"].sum()
    )
    if total_importance > 0:
        observations.append(
            f"The top three features account for {top_three_importance / total_importance:.1%} "
            "of the summed mean absolute SHAP values in this saved table."
        )
    return observations