from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS = [
    "Region",
    "Country",
    "Unit",
    "Key Account",
    "Type",
    "Product",
    "Status",
    "Condition",
    "Capacity (MT)",
    "Chassis Year",
    "Body Year",
    "Operation",
    "Engine Type",
    "Engine Number",
    "Chassis",
    "Axles",
    "LAT",
    "LON",
]

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_headers(df)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNS]

def load_fleet_excel(excel_path: str | Path, sheet_name: str = "Fleet") -> pd.DataFrame:
    excel_path = Path(excel_path)

    if not excel_path.exists():
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=COLUMNS)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return df

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError:
        df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")

    return ensure_columns(df)

def save_fleet_excel(df: pd.DataFrame, excel_path: str | Path, sheet_name: str = "Fleet") -> None:
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df = ensure_columns(df)

    # Excel must be closed to write, otherwise PermissionError.
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def update_vehicle_by_unit(df: pd.DataFrame, unit: str, updates: dict) -> pd.DataFrame:
    df = ensure_columns(df)
    idx = df.index[df["UNIT"].astype(str) == str(unit)]
    if len(idx) == 0:
        raise ValueError("Vehicle not found.")
    i = idx[0]
    for k, v in updates.items():
        if k in df.columns:
            df.at[i, k] = v
    return df

