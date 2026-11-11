"""Domain-oriented checks for the applicant assessment form."""

import math

import pandas as pd


def validate_form_values(
    *,
    age: int,
    annual_income: float,
    employment_length: float | None,
    loan_amount: float,
    interest_rate: float | None,
    credit_history_length: int,
) -> list[str]:
    """Return actionable validation messages for cross-field applicant inputs."""
    errors: list[str] = []
    numeric_values = {
        "age": age,
        "annual income": annual_income,
        "loan amount": loan_amount,
        "credit history length": credit_history_length,
    }
    if employment_length is not None:
        numeric_values["employment length"] = employment_length
    if interest_rate is not None:
        numeric_values["interest rate"] = interest_rate
    non_finite = [name for name, value in numeric_values.items() if not math.isfinite(value)]
    if non_finite:
        errors.extend(f"{name.capitalize()} must be a finite number." for name in non_finite)
        return errors

    if age < 18 or age > 100:
        errors.append("Age must be between 18 and 100 years.")
    if annual_income <= 0:
        errors.append("Annual income must be greater than zero.")
    if loan_amount <= 0:
        errors.append("Loan amount must be greater than zero.")
    if employment_length is not None:
        if employment_length < 0:
            errors.append("Employment length cannot be negative.")
        elif employment_length > age:
            errors.append("Employment length cannot exceed age.")
    if interest_rate is not None and interest_rate < 0:
        errors.append("Interest rate cannot be negative.")
    if credit_history_length < 0 or credit_history_length > age:
        errors.append("Credit history length must be between zero and the applicant's age.")
    if annual_income > 0 and loan_amount > 0:
        ratio = loan_amount / annual_income
        if not pd.notna(ratio) or ratio <= 0:
            errors.append("The derived loan-to-income ratio must be greater than zero.")
    return errors
