"""CreditWise AI Streamlit application foundation."""

import pandas as pd
import streamlit as st

from src.analytics import (
    filter_portfolio,
    filter_portfolio_analytics,
    portfolio_insights,
    portfolio_kpis,
    portfolio_summary_table,
)
from src.batch_assessment_view import render_batch_assessment
from src.data_loader import load_portfolio_data
from src.explainability_view import render_explainable_ai_page
from src.model_service import (
    get_model_schema,
    load_candidate_model,
    predict_default_probability,
    validate_applicant,
)
from src.validation import validate_form_values
from src.visualizations import (
    categorical_default_rate,
    age_default_rate,
    application_volume_by_purpose,
    income_default_rate,
    income_quartile_default_rate_figure,
    interest_rate_by_outcome,
    loan_amount_distribution,
    loan_amount_income_scatter,
    numerical_correlation_heatmap,
    ownership_outcomes,
    target_distribution,
)

st.set_page_config(
    page_title="CreditWise AI | Credit Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)

NAV_ITEMS = [
    "Executive Dashboard",
    "Applicant Risk Assessment",
    "Portfolio Analytics",
    "Explainable AI",
    "Batch Assessment",
]

st.markdown(
    """
    <style>
    :root {
        --cw-navy: #102A43;
        --cw-blue: #2563EB;
        --cw-teal: #087F7A;
        --cw-ink: #243B53;
        --cw-muted: #627D98;
        --cw-border: #E5EDF4;
        --cw-surface: #FFFFFF;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(34, 184, 207, .08), transparent 28%),
            #F4F7FB;
        color: var(--cw-ink);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102A43 0%, #163C5B 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #EAF4FF; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: .22rem .25rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.08);
    }
    .cw-brand {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1.36rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.04em;
        color: white;
        margin: .2rem 0 .25rem;
    }
    .cw-brand span { color: #39D2C0; }
    .cw-tagline {
        color: #B9D2E7;
        font-size: .79rem;
        line-height: 1.5;
        margin: 0 0 1.35rem;
    }
    .cw-eyebrow {
        color: var(--cw-teal);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .cw-title {
        color: var(--cw-navy);
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: clamp(1.7rem, 3vw, 2.45rem);
        font-weight: 800;
        letter-spacing: -.045em;
        line-height: 1.15;
        margin: 0;
    }
    .cw-subtitle {
        color: var(--cw-muted);
        font-size: 1rem;
        margin: .45rem 0 1.4rem;
    }
    .cw-panel {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FBFE 100%);
        border: 1px solid var(--cw-border);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin: .4rem 0 1rem;
        box-shadow: 0 8px 26px rgba(16, 42, 67, .045);
    }
    .cw-panel-title {
        color: var(--cw-navy);
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: .3rem;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--cw-border);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 6px 20px rgba(16, 42, 67, .04);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--cw-muted);
        font-size: .82rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: var(--cw-navy);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid var(--cw-border);
        border-radius: 16px;
        padding: 1rem 1.15rem .5rem;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(105deg, #2563EB, #087F7A);
        color: white;
        border: 0;
        border-radius: 10px;
        min-height: 2.7rem;
        font-weight: 700;
        transition: transform .15s ease, box-shadow .15s ease;
        box-shadow: 0 7px 16px rgba(37, 99, 235, .18);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(37, 99, 235, .25);
        border: 0;
    }
    .cw-footer {
        color: #526B83;
        font-size: .76rem;
        border-top: 1px solid var(--cw-border);
        padding-top: .8rem;
        margin-top: 2rem;
    }
    :focus-visible {
        outline: 3px solid #A64B00 !important;
        outline-offset: 2px;
    }
    .stButton > button, .stFormSubmitButton > button {
        min-height: 2.75rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        min-height: 2.75rem;
        display: flex;
        align-items: center;
    }
    [data-testid="stDataFrame"] {
        max-width: 100%;
    }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }
        [data-testid="stExpandSidebarButton"],
        button[aria-label="Collapse sidebar"],
        [data-testid="stFileUploader"] button,
        [data-testid="stTooltipIcon"] button {
            min-width: 2.75rem !important;
            min-height: 2.75rem !important;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: .75rem;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: min(100%, 19rem);
        }
        [data-testid="stMultiSelectTagsContainer"] {
            display: flex !important;
            max-width: 100%;
            height: auto !important;
            flex-wrap: wrap !important;
            overflow: visible !important;
        }
        [data-testid="stMultiSelectTagsContainer"] [aria-label="Selected values"] {
            display: flex !important;
            max-width: 100%;
            flex-wrap: wrap !important;
            gap: .15rem;
        }
        [data-testid="stMultiSelectTagsContainer"] [data-tag] {
            max-width: calc(100% - .25rem);
        }
        [data-testid="stMultiSelectTagsContainer"] [data-tag] > span {
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        [data-testid="stMultiSelect"] [role="combobox"] {
            min-height: 2.75rem;
        }
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem .875rem 1.5rem;
        }
        .cw-title {
            font-size: 1.75rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .cw-subtitle {
            font-size: .95rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }
        .cw-eyebrow {
            font-size: .7rem;
        }
        .cw-panel {
            border-radius: 12px;
            padding: .85rem;
        }
        [data-testid="stHorizontalBlock"] {
            gap: .5rem;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        div[data-testid="stMetric"] {
            min-height: 4.25rem;
            padding: .75rem .85rem;
            border-radius: 10px;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: .875rem;
            line-height: 1.3;
            white-space: normal;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem;
            overflow-wrap: anywhere;
        }
        [data-testid="stMultiSelectTagsContainer"] {
            display: flex !important;
            max-width: 100%;
            height: auto !important;
            flex-wrap: wrap !important;
            overflow: visible !important;
        }
        [data-testid="stMultiSelectTagsContainer"] [aria-label="Selected values"] {
            display: flex !important;
            max-width: 100%;
            flex-wrap: wrap !important;
            gap: .15rem;
        }
        div[data-testid="stMultiSelect"] [role="combobox"] {
            height: auto !important;
            min-height: 2.75rem;
            padding-top: .2rem;
            padding-bottom: .2rem;
        }
        [data-testid="stMultiSelectTagsContainer"] [data-tag] {
            max-width: calc(100% - .25rem);
        }
        [data-testid="stMultiSelectTagsContainer"] [data-tag] > span {
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .stButton > button, .stFormSubmitButton > button,
        [data-testid="stDownloadButton"] > button {
            width: 100%;
            min-height: 2.9rem;
            white-space: normal;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stMultiSelect"] [role="combobox"] {
            min-height: 2.75rem;
        }
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {
            min-height: 2.75rem;
            display: flex;
            align-items: center;
        }
        [data-testid="stFileUploader"] section {
            padding: .75rem;
        }
        [data-testid="stDataFrame"] {
            overflow-x: auto;
        }
        .cw-footer {
            font-size: .8rem;
            line-height: 1.45;
            margin-top: 1.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header(section_title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="cw-eyebrow">CreditWise AI &nbsp;·&nbsp; Phase 05</div>
        <h1 class="cw-title">{section_title}</h1>
        <p class="cw-subtitle">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(frame: pd.DataFrame) -> None:
    render_header(
        "Executive Dashboard",
        "A clear view of historical credit applications, portfolio mix, and observed outcomes.",
    )
    st.markdown(
        '<div class="cw-panel"><div class="cw-panel-title">Research dataset</div>'
        "This dashboard summarizes the cleaned historical dataset. Observed associations are descriptive, "
        "not causal, and do not establish future performance.</div>",
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        purposes = sorted(frame["loan_intent"].dropna().unique().tolist())
        selected_purposes = st.multiselect(
            "Loan purpose",
            options=purposes,
            default=purposes,
            help="Clear all selections to show the full portfolio.",
            key="dashboard_loan_purpose",
        )
    with filter_col2:
        ownerships = sorted(frame["person_home_ownership"].dropna().unique().tolist())
        selected_ownerships = st.multiselect(
            "Home ownership",
            options=ownerships,
            default=ownerships,
            help="Clear all selections to show the full portfolio.",
            key="dashboard_home_ownership",
        )

    filtered = filter_portfolio(frame, selected_purposes, selected_ownerships)
    if filtered.empty:
        st.warning("No applications match these filters. Select at least one available category.")
        return

    kpis = portfolio_kpis(filtered)
    metric_cols = st.columns(6)
    metric_cols[0].metric("Applications", f"{kpis['applications']:,}")
    metric_cols[1].metric("Defaults", f"{kpis['defaults']:,}")
    metric_cols[2].metric("Non-defaults", f"{kpis['non_defaults']:,}")
    metric_cols[3].metric("Observed default rate", f"{kpis['default_rate']:.1%}")
    metric_cols[4].metric("Average loan amount", f"${kpis['average_loan_amount']:,.0f}")
    metric_cols[5].metric("Average annual income", f"${kpis['average_income']:,.0f}")

    if "loan_grade" in filtered.columns:
        st.caption(
            "Loan grade is shown here for historical portfolio analysis only. It is not an input "
            "to the provisional prediction model and may reflect a lender's original assessment."
        )

    first_row = st.columns(2, gap="large")
    with first_row[0]:
        st.plotly_chart(
            target_distribution(filtered),
            width="stretch",
            config={"displayModeBar": False},
        )
    with first_row[1]:
        st.plotly_chart(
            categorical_default_rate(
                filtered,
                "loan_grade",
                "Observed default rate by loan grade",
                color="#2563EB",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    second_row = st.columns(2, gap="large")
    with second_row[0]:
        st.plotly_chart(
            categorical_default_rate(
                filtered,
                "loan_intent",
                "Observed default rate by loan purpose",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with second_row[1]:
        st.plotly_chart(
            income_default_rate(filtered),
            width="stretch",
            config={"displayModeBar": False},
        )

    third_row = st.columns(2, gap="large")
    with third_row[0]:
        st.plotly_chart(
            categorical_default_rate(
                filtered,
                "person_home_ownership",
                "Observed default rate by home ownership",
                color="#2563EB",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with third_row[1]:
        st.plotly_chart(
            ownership_outcomes(filtered),
            width="stretch",
            config={"displayModeBar": False},
        )
    st.caption(
        "Income quintiles are recalculated within the selected portfolio slice. Rates describe "
        "this dataset and should not be interpreted as causal effects or lending policy."
    )


def _reset_portfolio_analytics_filters() -> None:
    for key in (
        "portfolio_purpose",
        "portfolio_ownership",
        "portfolio_grade",
        "portfolio_age",
        "portfolio_income",
        "portfolio_loan_amount",
        "portfolio_employment",
        "portfolio_include_missing_employment",
    ):
        st.session_state.pop(key, None)


def render_portfolio_analytics(frame: pd.DataFrame) -> None:
    render_header(
        "Portfolio Analytics",
        "Investigate historical default patterns and risk concentrations across the selected portfolio.",
    )
    st.markdown(
        '<div class="cw-panel"><div class="cw-panel-title">Historical portfolio view</div>'
        "Amounts are shown in original USD units. Rates and correlations are descriptive summaries "
        "of this dataset; they do not establish causal relationships or predict future outcomes.</div>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("#### Portfolio filters")
        st.button(
            "Reset filters",
            key="portfolio_reset_filters",
            on_click=_reset_portfolio_analytics_filters,
            help="Restore all categories, full numeric ranges, and include missing employment lengths.",
        )
        category_cols = st.columns(3, gap="large")
        purposes = sorted(frame["loan_intent"].dropna().unique().tolist())
        ownerships = sorted(frame["person_home_ownership"].dropna().unique().tolist())
        grades = sorted(frame["loan_grade"].dropna().unique().tolist())
        with category_cols[0]:
            selected_purposes = st.multiselect(
                "Loan purpose", purposes, default=purposes, key="portfolio_purpose"
            )
        with category_cols[1]:
            selected_ownerships = st.multiselect(
                "Home ownership", ownerships, default=ownerships, key="portfolio_ownership"
            )
        with category_cols[2]:
            selected_grades = st.multiselect(
                "Loan grade", grades, default=grades, key="portfolio_grade"
            )

        range_cols = st.columns(3, gap="large")
        age_min, age_max = int(frame["person_age"].min()), int(frame["person_age"].max())
        income_min, income_max = int(frame["person_income"].min()), int(frame["person_income"].max())
        amount_min, amount_max = int(frame["loan_amnt"].min()), int(frame["loan_amnt"].max())
        with range_cols[0]:
            age_range = st.slider(
                "Applicant age (years)", age_min, age_max, (age_min, age_max), key="portfolio_age"
            )
        with range_cols[1]:
            income_range = st.slider(
                "Annual income (USD)",
                income_min,
                income_max,
                (income_min, income_max),
                step=1_000,
                key="portfolio_income",
            )
        with range_cols[2]:
            loan_amount_range = st.slider(
                "Loan amount (USD)",
                amount_min,
                amount_max,
                (amount_min, amount_max),
                step=500,
                key="portfolio_loan_amount",
            )

        employment_min = float(frame["person_emp_length"].min())
        employment_max = float(frame["person_emp_length"].max())
        employment_col, missing_col = st.columns([2, 1], gap="large")
        with employment_col:
            employment_range = st.slider(
                "Employment length (years)",
                employment_min,
                employment_max,
                (employment_min, employment_max),
                step=0.5,
                key="portfolio_employment",
            )
        with missing_col:
            include_missing_employment = st.checkbox(
                "Include missing employment lengths",
                value=True,
                key="portfolio_include_missing_employment",
            )

    filtered = filter_portfolio_analytics(
        frame,
        loan_intents=selected_purposes,
        home_ownerships=selected_ownerships,
        loan_grades=selected_grades,
        age_range=age_range,
        income_range=income_range,
        loan_amount_range=loan_amount_range,
        employment_range=employment_range,
        include_missing_employment=include_missing_employment,
    )
    kpis = portfolio_kpis(filtered)
    first_metrics = st.columns(4, gap="medium")
    first_metrics[0].metric("Applications", f"{kpis['applications']:,}")
    first_metrics[1].metric("Defaults", f"{kpis['defaults']:,}")
    first_metrics[2].metric("Non-defaults", f"{kpis['non_defaults']:,}")
    first_metrics[3].metric("Observed default rate", f"{kpis['default_rate']:.1%}")
    second_metrics = st.columns(4, gap="medium")
    second_metrics[0].metric("Average annual income", f"${kpis['average_income']:,.0f} USD")
    second_metrics[1].metric("Average loan amount", f"${kpis['average_loan_amount']:,.0f} USD")
    second_metrics[2].metric("Average interest rate", f"{kpis['average_interest_rate']:.2f}%")
    second_metrics[3].metric("Average loan-to-income ratio", f"{kpis['average_loan_to_income']:.1%}")

    st.download_button(
        "Download filtered portfolio CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="creditwise_filtered_portfolio.csv",
        mime="text/csv",
        key="portfolio_download_filtered",
    )
    st.download_button(
        "Download portfolio summary CSV",
        data=portfolio_summary_table(filtered).to_csv(index=False).encode("utf-8"),
        file_name="creditwise_portfolio_summary.csv",
        mime="text/csv",
        key="portfolio_download_summary",
    )

    st.markdown("#### Business observations")
    insights = portfolio_insights(filtered)
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No observations are available because no applications match the current filters.")
    st.caption(
        "These are calculated historical comparisons within the selected records. Group rates can "
        "vary with application counts and must not be interpreted as causal effects."
    )

    st.markdown("#### Portfolio visualizations")
    if filtered.empty:
        st.warning("No applications match these filters. Reset the filters or broaden a selection.")
        return

    chart_rows = [
        (
            categorical_default_rate(
                filtered,
                "loan_grade",
                "Observed default rate by loan grade",
                color="#2563EB",
            ),
            categorical_default_rate(filtered, "loan_intent", "Observed default rate by loan purpose"),
        ),
        (
            categorical_default_rate(
                filtered,
                "person_home_ownership",
                "Observed default rate by home ownership",
                "#2563EB",
            ),
            income_quartile_default_rate_figure(filtered),
        ),
        (age_default_rate(filtered), loan_amount_income_scatter(filtered)),
        (interest_rate_by_outcome(filtered), application_volume_by_purpose(filtered)),
        (loan_amount_distribution(filtered), numerical_correlation_heatmap(filtered)),
    ]
    for left_figure, right_figure in chart_rows:
        left, right = st.columns(2, gap="large")
        with left:
            st.plotly_chart(left_figure, width="stretch", config={"displayModeBar": False})
        with right:
            st.plotly_chart(right_figure, width="stretch", config={"displayModeBar": False})


def render_applicant_assessment(
    frame: pd.DataFrame,
    model: object,
    model_features: list[str],
    categories: dict[str, list[str]],
) -> None:
    render_header(
        "Applicant Risk Assessment",
        "Explore the saved research model with a single hypothetical applicant profile.",
    )
    st.warning(
        "Demonstration only — this tool estimates a default probability. It does not approve, "
        "decline, price, or recommend a real loan."
    )
    st.markdown(
        '<div class="cw-panel"><div class="cw-panel-title">How to read the estimate</div>'
        "The value below is the saved five-fold Random Forest ensemble's isotonic-calibrated "
        "estimate for the dataset label <code>loan_status = 1</code>. It is not a lending "
        "threshold. The model was trained on historical data; calibration can saturate at 0% or "
        "100%, and an estimate is not a guarantee of an individual outcome.</div>",
        unsafe_allow_html=True,
    )

    age_default = int(round(float(frame["person_age"].median())))
    income_default = int(round(float(frame["person_income"].median())))
    amount_default = int(round(float(frame["loan_amnt"].median())))
    history_default = int(round(float(frame["cb_person_cred_hist_length"].median())))
    employment_median = float(frame["person_emp_length"].median())
    rate_median = float(frame["loan_int_rate"].median())

    with st.form("applicant_assessment_form"):
        st.markdown("#### Applicant and credit characteristics")
        left, right = st.columns(2, gap="large")
        with left:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=min(max(age_default, 18), 100),
                step=1,
            )
            annual_income = st.number_input(
                "Annual income ($)",
                min_value=0,
                value=max(income_default, 0),
                step=1_000,
                help="Enter a positive annual income amount.",
            )
            home_ownership = st.selectbox(
                "Home ownership",
                options=categories["person_home_ownership"],
                index=(
                    categories["person_home_ownership"].index("MORTGAGE")
                    if "MORTGAGE" in categories["person_home_ownership"]
                    else 0
                ),
            )
            employment_missing = st.checkbox(
                "Employment length not provided",
                help="The saved pipeline will use its training-fitted median imputer.",
            )
            employment_length = st.number_input(
                "Employment length (years)",
                min_value=0.0,
                max_value=float(age),
                value=min(max(employment_median, 0.0), float(age)),
                step=0.5,
                disabled=employment_missing,
            )
            loan_intent = st.selectbox("Loan purpose", options=categories["loan_intent"])
        with right:
            loan_amount = st.number_input(
                "Loan amount ($)",
                min_value=0,
                value=max(amount_default, 0),
                step=500,
                help="Enter a positive loan amount.",
            )
            rate_missing = st.checkbox(
                "Interest rate not provided",
                help="The saved pipeline will use its training-fitted median imputer.",
            )
            interest_rate = st.number_input(
                "Interest rate (%)",
                min_value=0.0,
                value=max(rate_median, 0.0),
                step=0.1,
                disabled=rate_missing,
            )
            prior_default = st.selectbox(
                "Prior default on file",
                options=categories["cb_person_default_on_file"],
            )
            credit_history_length = st.number_input(
                "Credit history length (years)",
                min_value=0,
                max_value=int(age),
                value=min(max(history_default, 0), int(age)),
                step=1,
            )
            loan_to_income = (
                float(loan_amount) / float(annual_income)
                if annual_income > 0
                else 0.0
            )
            st.metric("Derived loan-to-income ratio", f"{loan_to_income:.1%}")
            st.caption("Calculated as loan amount ÷ annual income, matching the model feature.")

        submitted = st.form_submit_button(
            "Estimate default probability",
            width="stretch",
        )

    if submitted:
        employment_value = None if employment_missing else float(employment_length)
        rate_value = None if rate_missing else float(interest_rate)
        validation_errors = validate_form_values(
            age=int(age),
            annual_income=float(annual_income),
            employment_length=employment_value,
            loan_amount=float(loan_amount),
            interest_rate=rate_value,
            credit_history_length=int(credit_history_length),
        )
        if validation_errors:
            for message in validation_errors:
                st.error(message)
            return

        if loan_to_income > 1:
            st.info(
                "The entered loan amount exceeds annual income. The estimate can still be "
                "computed, but this input is outside common observed loan-to-income values."
            )

        applicant = {
            "person_age": int(age),
            "person_income": float(annual_income),
            "person_home_ownership": home_ownership,
            "person_emp_length": employment_value,
            "loan_intent": loan_intent,
            "loan_amnt": float(loan_amount),
            "loan_int_rate": rate_value,
            "loan_percent_income": loan_to_income,
            "cb_person_default_on_file": prior_default,
            "cb_person_cred_hist_length": int(credit_history_length),
        }
        try:
            model_input = validate_applicant(applicant, model_features, categories)
            probability = predict_default_probability(model, model_input)
        except (TypeError, ValueError) as error:
            st.error(f"Could not assess this profile: {error}")
            return

        out_of_range = []
        for feature in (
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
        ):
            value = applicant[feature]
            if value is None or pd.isna(value):
                continue
            minimum = float(frame[feature].min())
            maximum = float(frame[feature].max())
            if value < minimum or value > maximum:
                out_of_range.append(
                    f"{feature} ({value:g}; cleaned-data range {minimum:g}–{maximum:g})"
                )
        if out_of_range:
            st.warning(
                "Some inputs are outside the cleaned dataset's observed range. The model still "
                "returns an estimate, but it may be less reliable for these values: "
                + ", ".join(out_of_range)
            )

        st.markdown("---")
        st.markdown("### Estimated result")
        result_cols = st.columns([1, 2], gap="large")
        with result_cols[0]:
            st.metric(
                "Estimated default probability",
                f"{probability:.2%}",
                help="Calibrated model probability for the dataset's default class (loan_status = 1).",
            )
        with result_cols[1]:
            st.progress(
                probability,
                text="Estimated probability of the historical default label (not an approval score)",
            )
            st.caption(
                f"The complementary model estimate for non-default is {1 - probability:.2%}. "
                "Neither output is a decision or a guarantee."
            )

        profile = pd.DataFrame(
            [
                {
                    "Age": f"{int(age)} years",
                    "Annual income": f"${annual_income:,.0f}",
                    "Home ownership": home_ownership,
                    "Employment length": (
                        "Not provided" if employment_missing else f"{employment_length:g} years"
                    ),
                    "Loan purpose": loan_intent,
                    "Loan amount": f"${loan_amount:,.0f}",
                    "Interest rate": (
                        "Not provided" if rate_missing else f"{interest_rate:.2f}%"
                    ),
                    "Loan-to-income ratio": f"{loan_to_income:.1%}",
                    "Prior default on file": prior_default,
                    "Credit history": f"{int(credit_history_length)} years",
                }
            ]
        )
        st.markdown("#### Assessed characteristics")
        st.dataframe(profile, hide_index=True, width="stretch")
        st.caption(
            "No loan-grade input is used. Missing employment length or interest rate is handled "
            "by the candidate pipeline's fitted preprocessing. Saved example SHAP explanations "
            "are available in the separate Explainable AI section."
        )


def main() -> None:
    with st.sidebar:
        st.markdown('<div class="cw-brand">CreditWise <span>AI</span></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="cw-tagline">Intelligent Credit Risk Assessment &amp; Portfolio Analytics</p>',
            unsafe_allow_html=True,
        )
        st.markdown("##### WORKSPACE")
        selected_section = st.radio(
            "Application sections",
            options=NAV_ITEMS,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Research prototype · Phase 05")
        st.caption("Historical data is not a lending policy.")

    portfolio = load_portfolio_data()

    if selected_section == "Executive Dashboard":
        render_dashboard(portfolio)
    elif selected_section == "Applicant Risk Assessment":
        model = load_candidate_model()
        model_features, model_categories = get_model_schema(model)
        render_applicant_assessment(
            portfolio,
            model,
            model_features,
            model_categories,
        )
    elif selected_section == "Portfolio Analytics":
        render_portfolio_analytics(portfolio)
    elif selected_section == "Explainable AI":
        render_explainable_ai_page(render_header)
    elif selected_section == "Batch Assessment":
        try:
            model = load_candidate_model()
            model_features, model_categories = get_model_schema(model)
        except Exception as error:
            render_header(
                "Batch Assessment",
                "Validate multiple applicant records and estimate default probabilities with the saved model.",
            )
            st.error(f"Could not load the saved candidate model or its schema: {error}")
        else:
            render_batch_assessment(
                render_header,
                model,
                model_features,
                model_categories,
            )
    else:
        render_header(
            selected_section,
            "This section is planned for the next CreditWise AI implementation step.",
        )
        st.info(
            "The application foundation is in place. Batch Assessment is intentionally not "
            "implemented in this increment."
        )

    st.markdown(
        '<div class="cw-footer">CreditWise AI · Educational research prototype · '
        "Not financial, lending, or credit advice.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
