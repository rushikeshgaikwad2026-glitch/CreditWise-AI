"""Streamlit presentation for the saved Phase 4 SHAP artifacts."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from src.explainability import (
    GLOBAL_IMAGE_PATHS,
    PROFILE_IMAGE_PATHS,
    PROFILE_ORDER,
    display_feature_name,
    global_importance_observations,
    load_applicant_explanations,
    load_global_importance,
    load_png_artifact,
)


def _show_artifact_error(message: str | None) -> None:
    if message:
        st.warning(message)


def _contribution_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Feature": item["display_name"],
                "Observed value": item["observed_value"],
                "SHAP contribution (raw probability units)": float(item["shap_contribution"]),
            }
            for item in items
        ]
    )


def _profile_snapshot(profile: dict[str, Any]) -> pd.DataFrame:
    observed = profile["observed_features"]
    values = []
    for label, key, formatter in (
        ("Age", "person_age", lambda value: f"{float(value):.0f} years"),
        ("Annual income", "person_income", lambda value: f"${float(value):,.0f} USD"),
        ("Home ownership", "person_home_ownership", str),
        ("Employment length", "person_emp_length", lambda value: f"{float(value):.1f} years"),
        ("Loan purpose", "loan_intent", str),
        ("Loan amount", "loan_amnt", lambda value: f"${float(value):,.0f} USD"),
        ("Interest rate", "loan_int_rate", lambda value: f"{float(value):.2f}%"),
        ("Loan-to-income ratio", "loan_percent_income", lambda value: f"{float(value):.1%}"),
    ):
        value = observed.get(key)
        values.append({"Characteristic": label, "Saved value": formatter(value) if value is not None else "Not available"})
    return pd.DataFrame(values)


def _profile_interpretation(profile: dict[str, Any]) -> str:
    positive = profile["top_positive_contributions"]
    negative = profile["top_negative_contributions"]
    if positive and negative:
        return (
            f"For this saved profile, {positive[0]['display_name']} has the largest upward "
            f"attribution ({float(positive[0]['shap_contribution']):+.4f}), while "
            f"{negative[0]['display_name']} has the largest downward attribution "
            f"({float(negative[0]['shap_contribution']):+.4f}) in the first fold forest. "
            "These are model attributions for this profile, not causal effects."
        )
    return "The saved explanation does not contain both upward and downward contributions for this profile."


def render_explainable_ai_page(
    render_header: Callable[[str, str], None],
) -> None:
    """Render the Explainable AI page from existing saved artifacts only."""
    render_header(
        "Explainable AI",
        "Review what the saved research model learned and how its example explanations should be read.",
    )
    st.warning(
        "Critical distinction: these saved SHAP values explain the class-1 raw probability output "
        "of the first fold-specific Random Forest. They do not mathematically decompose the full "
        "isotonic-calibrated ensemble probability shown by Applicant Risk Assessment."
    )

    st.markdown("#### Understanding model behaviour")
    concept_cols = st.columns(3, gap="large")
    with concept_cols[0]:
        st.markdown("**Explainable AI**")
        st.write(
            "Here, it means examining which saved model inputs were associated with its estimated "
            "default-risk output, globally and for selected example records."
        )
    with concept_cols[1]:
        st.markdown("**Why it matters**")
        st.write(
            "Understanding model behaviour helps teams review patterns, spot unexpected reliance "
            "on inputs, and ask better validation questions before any real-world use."
        )
    with concept_cols[2]:
        st.markdown("**Attribution is not causation**")
        st.write(
            "Correlation describes variables that move together. SHAP attributes a fitted model "
            "output across its inputs. Neither shows that changing an input would cause a different outcome."
        )

    st.markdown("#### Global feature importance")
    importance, importance_error = load_global_importance()
    if importance is None:
        _show_artifact_error(importance_error)
    else:
        top_ten = importance.head(10).copy()
        chart_col, table_col = st.columns([1.15, 0.85], gap="large")
        with chart_col:
            image, image_error = load_png_artifact(str(GLOBAL_IMAGE_PATHS["global_importance"]))
            if image is not None:
                st.image(image, caption="Saved mean absolute SHAP importance", width="stretch")
            else:
                _show_artifact_error(image_error)
        with table_col:
            top_ten["Feature"] = top_ten["feature"].map(display_feature_name)
            table = top_ten.rename(
                columns={"mean_absolute_shap_value": "Mean absolute SHAP value"}
            )[["Feature", "Mean absolute SHAP value"]]
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Mean absolute SHAP value": st.column_config.NumberColumn(format="%.5f")
                },
            )
        for observation in global_importance_observations(importance):
            st.markdown(f"- {observation}")
        st.caption(
            "Mean absolute SHAP values summarize attribution magnitude across the saved sample. "
            "They do not indicate direction or causal importance."
        )

    st.markdown("#### SHAP summary analysis")
    beeswarm, beeswarm_error = load_png_artifact(
        str(GLOBAL_IMAGE_PATHS["summary_beeswarm"])
    )
    if beeswarm is not None:
        st.image(beeswarm, caption="Saved SHAP summary beeswarm", width="stretch")
    else:
        _show_artifact_error(beeswarm_error)
    st.markdown(
        "Each dot represents one saved applicant observation for a feature. Its horizontal position "
        "shows the SHAP contribution: positive values push the first fold forest's raw class-1 output "
        "upward from its baseline; negative values push it downward. Dot colour represents the feature "
        "value. A feature can push in different directions for different observations because the forest "
        "learns conditional patterns. This plot does not explain the complete calibrated ensemble."
    )

    st.markdown("#### Applicant explanation explorer")
    profiles, profiles_error = load_applicant_explanations()
    if profiles is None:
        _show_artifact_error(profiles_error)
    else:
        profiles_by_name = {profile["profile"]: profile for profile in profiles}
        selected_profile_name = st.radio(
            "Saved example profile",
            options=PROFILE_ORDER,
            horizontal=True,
            key="xai_saved_profile",
        )
        profile = profiles_by_name[selected_profile_name]
        st.markdown(f"##### {selected_profile_name}")
        probability_col, raw_col = st.columns(2, gap="large")
        calibrated_probability = profile.get("calibrated_candidate_default_probability")
        raw_probability = profile.get("fold_1_underlying_rf_raw_default_probability")
        with probability_col:
            st.metric(
                "Saved calibrated ensemble estimate",
                f"{float(calibrated_probability):.2%}"
                if calibrated_probability is not None
                else "Not available in artifacts",
            )
        with raw_col:
            st.metric(
                "First-fold forest raw output",
                f"{float(raw_probability):.2%}"
                if raw_probability is not None
                else "Not available in artifacts",
            )
        st.caption(
            "The calibrated estimate is read from the saved profile record. The SHAP contributions "
            "below explain only the first-fold forest raw output; they must not be added to or used to "
            "reconstruct the calibrated estimate."
        )

        snapshot_col, waterfall_col = st.columns([0.85, 1.15], gap="large")
        with snapshot_col:
            st.dataframe(_profile_snapshot(profile), hide_index=True, width="stretch")
        with waterfall_col:
            waterfall, waterfall_error = load_png_artifact(
                str(PROFILE_IMAGE_PATHS[selected_profile_name])
            )
            if waterfall is not None:
                st.image(
                    waterfall,
                    caption=f"Saved first-fold forest SHAP waterfall: {selected_profile_name}",
                    width="stretch",
                )
            else:
                _show_artifact_error(waterfall_error)

        contribution_cols = st.columns(2, gap="large")
        with contribution_cols[0]:
            st.markdown("**Pushing the fold-model output higher**")
            st.dataframe(
                _contribution_table(profile["top_positive_contributions"]),
                hide_index=True,
                width="stretch",
                column_config={
                    "SHAP contribution (raw probability units)": st.column_config.NumberColumn(
                        format="%+.4f"
                    )
                },
            )
        with contribution_cols[1]:
            st.markdown("**Pushing the fold-model output lower**")
            st.dataframe(
                _contribution_table(profile["top_negative_contributions"]),
                hide_index=True,
                width="stretch",
                column_config={
                    "SHAP contribution (raw probability units)": st.column_config.NumberColumn(
                        format="%+.4f"
                    )
                },
            )
        st.info(_profile_interpretation(profile))

    st.markdown("#### Responsible AI and model limitations")
    st.markdown(
        "- This is a research and decision-support demonstration, not a real lending approval system.\n"
        "- The model estimates default probabilities; it does not approve or reject loans.\n"
        "- SHAP describes model attribution, not causation.\n"
        "- These saved SHAP plots explain one fold-specific forest, not the complete calibrated ensemble.\n"
        "- The model may reflect patterns and biases present in its historical training data.\n"
        "- Subgroup differences require further investigation; they are not, by themselves, proof of discrimination or fairness.\n"
        "- Model performance and explanations are not guarantees about an individual applicant."
    )