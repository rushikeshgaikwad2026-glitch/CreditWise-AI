"""Plotly figures for the executive dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics import (
    age_group_default_rate,
    default_rate_by,
    income_quartile_default_rate,
    income_quintile_default_rate,
)

NAVY = "#102A43"
BLUE = "#2563EB"
TEAL = "#0F9D9A"
CYAN = "#22B8CF"
PALETTE = ["#2563EB", "#0F9D9A", "#22B8CF", "#7C83FD", "#FFB547", "#A78BFA"]
LIGHT_GRID = "#E8EEF4"


def _style_figure(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 16, "color": NAVY}},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#425466"},
        margin={"l": 16, "r": 16, "t": 56, "b": 16},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(showgrid=False, linecolor=LIGHT_GRID)
    figure.update_yaxes(gridcolor=LIGHT_GRID, zeroline=False)
    return figure


def target_distribution(frame: pd.DataFrame) -> go.Figure:
    counts = frame["loan_status"].value_counts().reindex([0, 1], fill_value=0)
    figure = px.pie(
        values=counts.values,
        names=["Non-default", "Default"],
        hole=0.72,
        color_discrete_sequence=[BLUE, TEAL],
    )
    figure.update_traces(
        texttemplate="%{percent:.1%}",
        textposition="inside",
        hovertemplate="%{label}<br>Applications: %{value:,}<br>Share: %{percent:.1%}<extra></extra>",
    )
    return _style_figure(figure, "Application outcome mix")


def categorical_default_rate(
    frame: pd.DataFrame,
    column: str,
    title: str,
    color: str = TEAL,
) -> go.Figure:
    summary = default_rate_by(frame, column)
    summary["label"] = summary.apply(
        lambda row: f"{row['default_rate']:.1%} · n={int(row['applications']):,}",
        axis=1,
    )
    figure = px.bar(
        summary,
        x=column,
        y="default_rate",
        color_discrete_sequence=[color],
        text="label",
        custom_data=["applications", "defaults"],
    )
    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "%{x}<br>Default rate: %{y:.1%}<br>"
            "Applications: %{customdata[0]:,}<br>Defaults: %{customdata[1]:,}<extra></extra>"
        ),
    )
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    return _style_figure(figure, title)


def income_default_rate(frame: pd.DataFrame) -> go.Figure:
    summary = income_quintile_default_rate(frame)
    figure = px.bar(
        summary,
        x="income_quintile",
        y="default_rate",
        color_discrete_sequence=[BLUE],
        custom_data=["applications", "defaults"],
    )
    figure.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside",
        hovertemplate=(
            "%{x}<br>Default rate: %{y:.1%}<br>"
            "Applications: %{customdata[0]:,}<br>Defaults: %{customdata[1]:,}<extra></extra>"
        ),
    )
    figure.update_xaxes(title="Annual income quintile (slice-specific)")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    return _style_figure(figure, "Default rate by income quintile")


def income_quartile_default_rate_figure(frame: pd.DataFrame) -> go.Figure:
    summary = income_quartile_default_rate(frame)
    summary["label"] = summary.apply(
        lambda row: f"{row['default_rate']:.1%} · n={int(row['applications']):,}",
        axis=1,
    )
    figure = px.bar(
        summary,
        x="income_quartile",
        y="default_rate",
        color_discrete_sequence=[BLUE],
        text="label",
        custom_data=["applications", "defaults"],
    )
    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "%{x}<br>Default rate: %{y:.1%}<br>"
            "Applications: %{customdata[0]:,}<br>Defaults: %{customdata[1]:,}<extra></extra>"
        ),
    )
    figure.update_xaxes(title="Annual income quartile (filtered portfolio)")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    return _style_figure(figure, "Observed default rate by income quartile")


def age_default_rate(frame: pd.DataFrame) -> go.Figure:
    summary = age_group_default_rate(frame)
    summary["label"] = summary.apply(
        lambda row: f"{row['default_rate']:.1%} · n={int(row['applications']):,}",
        axis=1,
    )
    figure = px.bar(
        summary,
        x="age_group",
        y="default_rate",
        color_discrete_sequence=[TEAL],
        text="label",
        custom_data=["applications", "defaults"],
    )
    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "%{x}<br>Default rate: %{y:.1%}<br>"
            "Applications: %{customdata[0]:,}<br>Defaults: %{customdata[1]:,}<extra></extra>"
        ),
    )
    figure.update_xaxes(title="Applicant age group (years)")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    return _style_figure(figure, "Observed default rate by age group")


def loan_amount_income_scatter(frame: pd.DataFrame) -> go.Figure:
    plotted = frame.copy()
    plotted["outcome"] = plotted["loan_status"].map({0: "Non-default", 1: "Default"})
    figure = px.scatter(
        plotted,
        x="person_income",
        y="loan_amnt",
        color="outcome",
        color_discrete_map={"Non-default": BLUE, "Default": TEAL},
        opacity=0.58,
        hover_data={
            "person_age": True,
            "loan_int_rate": ":.2f",
            "loan_percent_income": ":.1%",
            "person_income": ":$,.0f",
            "loan_amnt": ":$,.0f",
            "outcome": True,
        },
    )
    figure.update_xaxes(title="Annual income (USD)", tickprefix="$", tickformat=",.0f")
    figure.update_yaxes(title="Loan amount (USD)", tickprefix="$", tickformat=",.0f")
    return _style_figure(figure, "Loan amount versus annual income")


def interest_rate_by_outcome(frame: pd.DataFrame) -> go.Figure:
    plotted = frame.loc[frame["loan_int_rate"].notna()].copy()
    plotted["outcome"] = plotted["loan_status"].map({0: "Non-default", 1: "Default"})
    figure = px.box(
        plotted,
        x="outcome",
        y="loan_int_rate",
        color="outcome",
        color_discrete_map={"Non-default": BLUE, "Default": TEAL},
        points="outliers",
    )
    figure.update_xaxes(title="Loan outcome")
    figure.update_yaxes(title="Interest rate (%)", ticksuffix="%")
    return _style_figure(figure, "Interest rate distribution by loan outcome")


def application_volume_by_purpose(frame: pd.DataFrame) -> go.Figure:
    counts = frame["loan_intent"].value_counts().rename_axis("loan_intent").reset_index(name="applications")
    figure = px.bar(
        counts,
        x="loan_intent",
        y="applications",
        color_discrete_sequence=[BLUE],
        text="applications",
    )
    figure.update_traces(texttemplate="%{text:,}", textposition="outside")
    figure.update_xaxes(title="Loan purpose")
    figure.update_yaxes(title="Applications", rangemode="tozero")
    return _style_figure(figure, "Loan application volume by purpose")


def loan_amount_distribution(frame: pd.DataFrame) -> go.Figure:
    plotted = frame.copy()
    plotted["outcome"] = plotted["loan_status"].map({0: "Non-default", 1: "Default"})
    figure = px.histogram(
        plotted,
        x="loan_amnt",
        color="outcome",
        barmode="overlay",
        opacity=0.68,
        color_discrete_map={"Non-default": BLUE, "Default": TEAL},
        nbins=35,
    )
    figure.update_xaxes(title="Loan amount (USD)", tickprefix="$", tickformat=",.0f")
    figure.update_yaxes(title="Applications", rangemode="tozero")
    return _style_figure(figure, "Loan amount distribution by default status")


def numerical_correlation_heatmap(frame: pd.DataFrame) -> go.Figure:
    numerical_columns = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_status",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    correlation = frame[numerical_columns].corr()
    figure = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, BLUE], [0.5, "#F4F7FB"], [1, TEAL]],
            text=correlation.round(2).values,
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar={"title": "Correlation"},
        )
    )
    return _style_figure(figure, "Numerical feature correlation (descriptive)")


def ownership_outcomes(frame: pd.DataFrame) -> go.Figure:
    summary = (
        frame.groupby(["person_home_ownership", "loan_status"], observed=True)
        .size()
        .rename("applications")
        .reset_index()
    )
    summary["outcome"] = summary["loan_status"].map({0: "Non-default", 1: "Default"})
    figure = px.bar(
        summary,
        x="person_home_ownership",
        y="applications",
        color="outcome",
        barmode="stack",
        color_discrete_map={"Non-default": BLUE, "Default": TEAL},
        category_orders={"outcome": ["Non-default", "Default"]},
    )
    figure.update_yaxes(title="Applications")
    return _style_figure(figure, "Applications by home ownership")
