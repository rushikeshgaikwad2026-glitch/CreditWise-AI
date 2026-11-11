"""Streamlit presentation for validating and assessing applicant CSV batches."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.batch_assessment import (
    build_sample_template,
    inspect_upload_columns,
    load_reference_threshold,
    risk_category_labels,
    run_batch_predictions,
)

BLUE = "#2563EB"
TEAL = "#0F9D9A"
NAVY = "#102A43"
GRID = "#E8EEF4"


def _style_batch_chart(figure: Any, title: str) -> Any:
    figure.update_layout(
        title={"text": title, "font": {"size": 16, "color": NAVY}},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#425466"},
        margin={"l": 16, "r": 16, "t": 56, "b": 16},
        showlegend=False,
    )
    figure.update_xaxes(showgrid=False, linecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zeroline=False)
    return figure


def _format_row_label(row_number: int, probability: float | None = None) -> str:
    label = f"CSV row {row_number}"
    if probability is not None:
        label += f" ({probability:.2%})"
    return label


def _compact_category_label(category: str) -> str:
    if "training-OOF reference" in category:
        return (
            "Below training reference"
            if category.startswith("Below ")
            else "At/above training reference"
        )
    return category.split(" (illustrative", maxsplit=1)[0]


def _render_batch_charts(
    results: pd.DataFrame,
    valid_count: int,
    invalid_count: int,
    categories: list[str],
) -> None:
    valid_results = results.loc[results["Estimated default probability"].notna()].copy()
    category_col, probability_col = st.columns(2, gap="large")
    with category_col:
        if valid_results.empty:
            st.info("Risk-category chart is unavailable until at least one applicant is assessed.")
        else:
            category_counts = (
                valid_results["Risk category"]
                .value_counts()
                .reindex(categories, fill_value=0)
                .rename_axis("Risk category")
                .reset_index(name="Applicants")
            )
            category_counts["Chart category"] = category_counts["Risk category"].map(
                _compact_category_label
            )
            figure = px.bar(
                category_counts,
                x="Applicants",
                y="Chart category",
                orientation="h",
                category_orders={
                    "Chart category": category_counts["Chart category"].tolist()
                },
                color_discrete_sequence=[BLUE, TEAL],
                text="Applicants",
            )
            figure.update_traces(textposition="outside")
            figure.update_xaxes(title="Applicants", rangemode="tozero")
            figure.update_yaxes(title=None)
            figure.update_layout(margin={"l": 140, "r": 20, "t": 56, "b": 20})
            st.plotly_chart(
                _style_batch_chart(figure, "Risk category distribution"),
                width="stretch",
                config={"displayModeBar": False},
            )
    with probability_col:
        if valid_results.empty:
            st.info("Probability distribution is unavailable until at least one applicant is assessed.")
        else:
            valid_results["Estimated default probability"] = pd.to_numeric(
                valid_results["Estimated default probability"]
            )
            figure = px.histogram(
                valid_results,
                x="Estimated default probability",
                nbins=20,
                color_discrete_sequence=[TEAL],
            )
            figure.update_xaxes(title="Estimated default probability", tickformat=".0%")
            figure.update_yaxes(title="Applicants", rangemode="tozero")
            st.plotly_chart(
                _style_batch_chart(figure, "Estimated probability distribution"),
                width="stretch",
                config={"displayModeBar": False},
            )

    top_col, status_col = st.columns(2, gap="large")
    with top_col:
        if valid_results.empty:
            st.info("Top-applicant chart is unavailable until at least one applicant is assessed.")
        else:
            top_applicants = valid_results.nlargest(
                10, "Estimated default probability"
            ).sort_values("Estimated default probability")
            top_applicants["CSV row"] = top_applicants["Original row number"].map(
                lambda row_number: f"Row {int(row_number)}"
            )
            figure = px.bar(
                top_applicants,
                x="Estimated default probability",
                y="CSV row",
                orientation="h",
                color_discrete_sequence=[BLUE],
                text="Estimated default probability",
            )
            figure.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            figure.update_xaxes(title="Estimated default probability", tickformat=".0%")
            figure.update_yaxes(title="Original CSV row")
            st.plotly_chart(
                _style_batch_chart(figure, "Top 10 estimated default probabilities"),
                width="stretch",
                config={"displayModeBar": False},
            )
    with status_col:
        status_data = pd.DataFrame(
            {
                "Record status": ["Valid and assessed", "Invalid / not assessed"],
                "Applicants": [valid_count, invalid_count],
            }
        )
        figure = px.bar(
            status_data,
            x="Record status",
            y="Applicants",
            color="Record status",
            color_discrete_map={"Valid and assessed": BLUE, "Invalid / not assessed": TEAL},
            text="Applicants",
        )
        figure.update_traces(textposition="outside")
        figure.update_yaxes(title="Applicants", rangemode="tozero")
        st.plotly_chart(
            _style_batch_chart(figure, "Valid versus invalid records"),
            width="stretch",
            config={"displayModeBar": False},
        )


def render_batch_assessment(
    render_header: Callable[[str, str], None],
    model: Any,
    model_features: list[str],
    categories: dict[str, list[str]],
) -> None:
    """Render the complete batch upload, validation, prediction, and reporting flow."""
    render_header(
        "Batch Assessment",
        "Validate multiple applicant records and estimate default probabilities with the saved model.",
    )
    st.warning(
        "Research and decision-support demonstration only. Results are estimated default "
        "probabilities, not loan approvals or rejections."
    )

    threshold, threshold_rule = load_reference_threshold()
    if threshold is not None:
        st.info(
            f"Risk categories are labeled relative to the saved {threshold:.0%} training "
            f"out-of-fold F1 reference threshold ({threshold_rule}). This is a reporting band, "
            "not a lending policy or a change to the model."
        )
    else:
        st.info(
            "The saved reference threshold is unavailable. Illustrative probability bands are "
            "used for reporting only; they are not lending policy."
        )

    st.markdown("#### Upload applicants")
    st.write(
        "Upload one CSV with the exact model input column names. The target column `loan_status` "
        "is not required and is never used for assessment. `person_emp_length` and `loan_int_rate` "
        "may be blank or omitted; the saved pipeline will handle those missing values. "
        "`loan_percent_income` is derived from loan amount divided by annual income; if supplied, "
        "it must agree with that calculation within rounding to two decimal places."
    )
    st.download_button(
        "Download sample CSV template",
        data=build_sample_template(model_features).to_csv(index=False).encode("utf-8"),
        file_name="creditwise_batch_template.csv",
        mime="text/csv",
        key="batch_template_download",
    )
    with st.expander("Required and optional columns"):
        required = inspect_upload_columns([], model_features)
        st.markdown("**Model input columns**")
        st.code(", ".join(model_features), language=None)
        st.markdown("**Required upload columns**")
        st.code(", ".join(required["missing_required"]), language=None)
        st.markdown("**Optional or derived columns**")
        st.code(", ".join(required["missing_optional"]), language=None)
        st.markdown("**Supported categorical values**")
        for field, values in categories.items():
            st.caption(f"{field}: {', '.join(values)}")

    uploaded_file = st.file_uploader(
        "Choose applicant CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="batch_applicant_upload",
        help="Rows are validated independently; invalid rows remain visible in the results and error report.",
    )
    if uploaded_file is None:
        st.info("Upload a CSV file to preview and assess applicant records.")
        return

    try:
        uploaded = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return

    st.markdown("#### Uploaded records")
    st.metric("Total uploaded applicants", f"{len(uploaded):,}")
    if uploaded.empty:
        st.warning("The uploaded CSV contains no applicant rows.")
        return
    st.dataframe(uploaded.head(10), hide_index=True, width="stretch")

    column_report = inspect_upload_columns(list(uploaded.columns), model_features)
    diagnostic_cols = st.columns(3, gap="large")
    diagnostic_cols[0].metric("Missing required columns", len(column_report["missing_required"]))
    diagnostic_cols[1].metric("Missing optional / derived columns", len(column_report["missing_optional"]))
    diagnostic_cols[2].metric("Unexpected columns", len(column_report["unexpected"]))
    if column_report["missing_required"]:
        st.error("Missing required columns: " + ", ".join(column_report["missing_required"]))
    if column_report["missing_optional"]:
        st.caption(
            "These columns may be omitted: " + ", ".join(column_report["missing_optional"])
        )
    if column_report["unexpected"]:
        st.warning(
            "Unexpected columns will be preserved in the report but ignored by the model: "
            + ", ".join(column_report["unexpected"])
        )
        if "loan_status" in column_report["unexpected"]:
            st.caption("The uploaded target column `loan_status` is ignored and is not used for prediction.")

    with st.spinner("Validating applicant rows and assessing valid records..."):
        batch = run_batch_predictions(
            uploaded,
            model,
            model_features,
            categories,
            threshold,
        )

    results = batch["results"]
    valid_count = batch["valid_count"]
    invalid_count = batch["invalid_count"]
    probabilities = pd.to_numeric(
        results["Estimated default probability"], errors="coerce"
    ).dropna()

    st.markdown("#### Batch summary")
    summary_cols = st.columns(4, gap="medium")
    summary_cols[0].metric("Total uploaded", f"{batch['total_records']:,}")
    summary_cols[1].metric("Valid applicants", f"{valid_count:,}")
    summary_cols[2].metric("Invalid applicants", f"{invalid_count:,}")
    summary_cols[3].metric(
        "Average estimated default probability",
        f"{probabilities.mean():.2%}" if not probabilities.empty else "N/A",
    )
    range_cols = st.columns(2, gap="medium")
    range_cols[0].metric(
        "Highest estimated probability",
        f"{probabilities.max():.2%}" if not probabilities.empty else "N/A",
    )
    range_cols[1].metric(
        "Lowest estimated probability",
        f"{probabilities.min():.2%}" if not probabilities.empty else "N/A",
    )

    category_counts = results.loc[
        results["Estimated default probability"].notna(), "Risk category"
    ].value_counts().reindex(risk_category_labels(threshold), fill_value=0)
    st.caption(
        "Risk category counts: "
        + "; ".join(f"{category}: {int(count):,}" for category, count in category_counts.items())
    )
    _render_batch_charts(
        results,
        valid_count,
        invalid_count,
        risk_category_labels(threshold),
    )

    st.markdown("#### Applicant result explorer")
    if batch["valid_records"]:
        valid_records = batch["valid_records"]
        record_by_row = {
            record["Original row number"]: record for record in valid_records
        }
        row_numbers = list(record_by_row)
        selected_row = st.selectbox(
            "Select an assessed applicant",
            options=row_numbers,
            format_func=lambda row_number: _format_row_label(
                row_number,
                record_by_row[row_number]["Estimated default probability"],
            ),
        )
        selected = record_by_row[selected_row]
        explorer_cols = st.columns(3, gap="medium")
        explorer_cols[0].metric(
            "Estimated default probability",
            f"{selected['Estimated default probability']:.2%}",
        )
        explorer_cols[1].metric("Risk category", selected["Risk category"])
        explorer_cols[2].metric("Validation status", "Valid and assessed")
        st.markdown("**Original uploaded input values**")
        original_values = selected["Original input"]
        st.dataframe(
            pd.DataFrame(
                [{"Field": field, "Uploaded value": value} for field, value in original_values.items()]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            "This probability is the saved calibrated model's estimate of the historical `loan_status = 1` "
            "label for the submitted input. It is not a guarantee or an approval decision. Batch assessment "
            "does not generate SHAP explanations for these arbitrary uploaded records."
        )
        for warning in selected["Validation warnings"]:
            st.warning(warning)
    else:
        st.info("No valid applicants are available to explore.")

    st.markdown("#### Records by assessment status")
    valid_rows = results.loc[results["Estimated default probability"].notna()]
    invalid_rows = results.loc[results["Estimated default probability"].isna()]
    valid_tab, invalid_tab = st.tabs(["Valid and assessed", "Invalid / not assessed"])
    with valid_tab:
        if valid_rows.empty:
            st.info("No valid applicants were assessed.")
        else:
            st.dataframe(valid_rows, hide_index=True, width="stretch")
    with invalid_tab:
        if invalid_rows.empty:
            st.success("No invalid applicant records.")
        else:
            st.dataframe(invalid_rows, hide_index=True, width="stretch")

    st.markdown("#### Assessment results")
    st.dataframe(results, hide_index=True, width="stretch")
    st.download_button(
        "Download batch assessment results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="creditwise_batch_assessment_results.csv",
        mime="text/csv",
        key="batch_results_download",
    )
    st.markdown("#### Validation errors")
    if batch["validation_errors"].empty:
        st.success("No invalid applicant rows were found.")
    else:
        st.dataframe(batch["validation_errors"], hide_index=True, width="stretch")
    st.download_button(
        "Download validation errors CSV",
        data=batch["validation_errors"].to_csv(index=False).encode("utf-8"),
        file_name="creditwise_batch_validation_errors.csv",
        mime="text/csv",
        key="batch_errors_download",
    )