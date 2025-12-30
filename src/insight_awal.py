import pandas as pd

def build_insights(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include="number")

    corr = None
    if not num.empty:
        corr = num.corr(numeric_only=True)

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "corr": corr,
    }
