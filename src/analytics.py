"""Portfolio summaries used by the executive dashboard."""

import pandas as pd


def filter_portfolio(
    frame: pd.DataFrame,
    loan_intents: list[str],
    home_ownerships: list[str],
) -> pd.DataFrame:
    """Filter the portfolio by selected purpose and home-ownership categories."""
    filtered = frame
    if loan_intents:
        filtered = filtered[filtered["loan_intent"].isin(loan_intents)]
    if home_ownerships:
        filtered = filtered[filtered["person_home_ownership"].isin(home_ownerships)]
    return filtered.copy()


def portfolio_kpis(frame: pd.DataFrame) -> dict[str, float | int]:
    """Calculate headline counts and averages for a portfolio slice."""
    defaults = int(frame["loan_status"].sum())
    applications = int(len(frame))
    return {
        "applications": applications,
        "defaults": defaults,
        "non_defaults": applications - defaults,
        "default_rate": defaults / applications if applications else 0.0,
        "average_loan_amount": float(frame["loan_amnt"].mean()) if applications else 0.0,
        "average_income": float(frame["person_income"].mean()) if applications else 0.0,
        "average_interest_rate": (
            float(frame["loan_int_rate"].mean())
            if applications and frame["loan_int_rate"].notna().any()
            else 0.0
        ),
        "average_loan_to_income": (
            float(frame["loan_percent_income"].mean()) if applications else 0.0
        ),
    }


def filter_portfolio_analytics(
    frame: pd.DataFrame,
    *,
    loan_intents: list[str],
    home_ownerships: list[str],
    loan_grades: list[str],
    age_range: tuple[int, int],
    income_range: tuple[int, int],
    loan_amount_range: tuple[int, int],
    employment_range: tuple[float, float],
    include_missing_employment: bool = True,
) -> pd.DataFrame:
    """Apply the interactive Portfolio Analytics filters as one portfolio slice."""
    filtered = frame[
        frame["loan_intent"].isin(loan_intents)
        & frame["person_home_ownership"].isin(home_ownerships)
        & frame["loan_grade"].isin(loan_grades)
        & frame["person_age"].between(*age_range)
        & frame["person_income"].between(*income_range)
        & frame["loan_amnt"].between(*loan_amount_range)
    ]
    employment_matches = frame["person_emp_length"].between(*employment_range)
    if include_missing_employment:
        employment_matches |= frame["person_emp_length"].isna()
    return filtered.loc[employment_matches.loc[filtered.index]].copy()


def default_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate application count, default count, and default rate by a field."""
    grouped = (
        frame.groupby(column, observed=True, dropna=False)["loan_status"]
        .agg(applications="size", defaults="sum", default_rate="mean")
        .reset_index()
    )
    return grouped.sort_values(column, kind="stable").reset_index(drop=True)


def income_quintile_default_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize default rates across income quintiles calculated from this slice."""
    result = frame.loc[:, ["person_income", "loan_status"]].copy()
    result["income_quintile"] = pd.qcut(
        result["person_income"],
        q=5,
        duplicates="drop",
    )
    summary = (
        result.groupby("income_quintile", observed=True)["loan_status"]
        .agg(applications="size", defaults="sum", default_rate="mean")
        .reset_index()
    )
    summary["income_quintile"] = summary["income_quintile"].astype(str)
    return summary


def income_quartile_default_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize default rates by income quartile within the supplied slice."""
    columns = ["income_quartile", "applications", "defaults", "default_rate"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    result = frame.loc[:, ["person_income", "loan_status"]].copy()
    quartile_count = min(4, len(result))
    labels = [f"Q{index}" for index in range(1, quartile_count + 1)]
    labels[0] = "Q1 (lowest)"
    if quartile_count > 1:
        labels[-1] = f"Q{quartile_count} (highest)"
    else:
        labels[0] = "Q1 (only segment)"
    ranks = result["person_income"].rank(method="first")
    result["income_quartile"] = pd.qcut(ranks, q=quartile_count, labels=labels)
    summary = (
        result.groupby("income_quartile", observed=True)["loan_status"]
        .agg(applications="size", defaults="sum", default_rate="mean")
        .reset_index()
    )
    summary["income_quartile"] = summary["income_quartile"].astype(str)
    return summary[columns]


def age_group_default_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize default rates across stable age bands."""
    grouped = frame.loc[:, ["person_age", "loan_status"]].copy()
    grouped["age_group"] = pd.cut(
        grouped["person_age"],
        bins=[0, 25, 35, 45, 55, 65, float("inf")],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
    )
    summary = (
        grouped.groupby("age_group", observed=True)["loan_status"]
        .agg(applications="size", defaults="sum", default_rate="mean")
        .reset_index()
    )
    summary["age_group"] = summary["age_group"].astype(str)
    return summary


def portfolio_insights(frame: pd.DataFrame) -> list[str]:
    """Calculate descriptive risk observations for the currently selected slice."""
    if frame.empty:
        return []

    observations: list[str] = []
    for column, label in (
        ("loan_intent", "Loan purpose"),
        ("person_home_ownership", "Home ownership"),
    ):
        summary = default_rate_by(frame, column)
        highest = summary.loc[summary["default_rate"].idxmax()]
        observations.append(
            f"{label}: {highest[column]} has the highest observed default rate "
            f"({highest['default_rate']:.1%}; n={int(highest['applications']):,})."
        )

    income_summary = income_quartile_default_rate(frame)
    if not income_summary.empty:
        highest_income = income_summary.loc[income_summary["default_rate"].idxmax()]
        observations.append(
            f"Income segment: {highest_income['income_quartile']} has the highest observed "
            f"default rate ({highest_income['default_rate']:.1%}; "
            f"n={int(highest_income['applications']):,})."
        )

    status_income = frame.groupby("loan_status")["loan_percent_income"].mean()
    default_ratio = float(status_income.get(1, float("nan")))
    non_default_ratio = float(status_income.get(0, float("nan")))
    if pd.notna(default_ratio) and pd.notna(non_default_ratio):
        difference = default_ratio - non_default_ratio
        direction = "higher" if difference >= 0 else "lower"
        observations.append(
            f"Average loan-to-income ratio: defaults average {default_ratio:.1%} versus "
            f"{non_default_ratio:.1%} for non-defaults ({abs(difference):.1%} {direction})."
        )
    else:
        observations.append(
            "Average loan-to-income ratio comparison requires both default and non-default records."
        )
    return observations


def portfolio_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a downloadable KPI and segment summary for a filtered portfolio."""
    records: list[dict[str, object]] = []
    kpis = portfolio_kpis(frame)
    for metric, value in kpis.items():
        records.append(
            {
                "summary_type": "portfolio_kpi",
                "segment": metric,
                "applications": kpis["applications"],
                "defaults": kpis["defaults"],
                "default_rate": kpis["default_rate"],
                "value": value,
            }
        )

    if not frame.empty:
        breakdowns = [
            ("loan_grade", default_rate_by(frame, "loan_grade")),
            ("loan_purpose", default_rate_by(frame, "loan_intent")),
            ("home_ownership", default_rate_by(frame, "person_home_ownership")),
            ("income_quartile", income_quartile_default_rate(frame)),
            ("age_group", age_group_default_rate(frame)),
        ]
        for dimension, summary in breakdowns:
            segment_column = next(
                column
                for column in summary.columns
                if column not in {"applications", "defaults", "default_rate"}
            )
            for row in summary.to_dict(orient="records"):
                records.append(
                    {
                        "summary_type": dimension,
                        "segment": row[segment_column],
                        "applications": int(row["applications"]),
                        "defaults": int(row["defaults"]),
                        "default_rate": float(row["default_rate"]),
                        "value": None,
                    }
                )

    return pd.DataFrame(
        records,
        columns=["summary_type", "segment", "applications", "defaults", "default_rate", "value"],
    )
