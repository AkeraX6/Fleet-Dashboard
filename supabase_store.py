"""
Supabase data access layer for Fleet Dashboard.
Handles reading/writing fleet data to Supabase Postgres.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import create_client, Client

# App column names (Excel-style with spaces) -> DB column names (underscore style)
APP_TO_DB = {
    "Region": "region",
    "Country": "country",
    "Unit": "unit",
    "Key Account": "key_account",
    "Type": "type",
    "Product": "product",
    "Status": "status",
    "Condition": "condition",
    "Capacity (MT)": "capacity_mt",
    "Chassis Year": "chassis_year",
    "Body Year": "body_year",
    "Operation": "operation",
    "Engine Type": "engine_type",
    "Engine Number": "engine_number",
    "Chassis": "chassis",
    "Axles": "axles",
    "LAT": "lat",
    "LON": "lon",
}

# Reverse mapping: DB -> App
DB_TO_APP = {v: k for k, v in APP_TO_DB.items()}

# Table name in Supabase
TABLE_NAME = "fleet"


def get_supabase_client() -> Client:
    """
    Create and return a Supabase client using Streamlit secrets.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def has_supabase_secrets() -> bool:
    """
    Check if Supabase secrets are configured.
    Returns False if secrets are missing (for local dev fallback).
    """
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        return bool(url) and bool(key)
    except Exception:
        return False


def load_fleet_db(fleet_type: str, app_columns: list[str]) -> pd.DataFrame:
    """
    Load fleet data from Supabase filtered by fleet_type.
    Returns a DataFrame with app column names (spaces preserved).
    
    Args:
        fleet_type: "OP" or "UG"
        app_columns: list of expected column names (from utils.COLUMNS)
    
    Returns:
        DataFrame with app column names in correct order
    """
    sb = get_supabase_client()
    
    response = (
        sb.table(TABLE_NAME)
        .select("*")
        .eq("fleet_type", fleet_type)
        .order("unit")
        .execute()
    )
    
    rows = response.data or []
    
    if not rows:
        return pd.DataFrame(columns=app_columns)
    
    df = pd.DataFrame(rows)
    
    # Drop DB-only columns that don't map to app columns
    for drop_col in ["fleet_type", "updated_at", "id"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
    
    # Rename DB columns to app column names
    df = df.rename(columns=DB_TO_APP)
    
    # Ensure all expected columns exist and in correct order
    for col in app_columns:
        if col not in df.columns:
            df[col] = pd.NA
    
    return df[app_columns]


def upsert_vehicle_db(fleet_type: str, row_app: dict) -> None:
    """
    Upsert a vehicle row in Supabase.
    Uses (fleet_type, unit) as the conflict key.
    
    Args:
        fleet_type: "OP" or "UG"
        row_app: dict with app column names (e.g., {"Unit": "UH07", "Region": "AFRICA", ...})
    """
    sb = get_supabase_client()
    
    unit = str(row_app.get("Unit", "")).strip()
    if not unit:
        raise ValueError("Unit is required")
    
    # Build payload with DB column names
    payload = {
        "fleet_type": fleet_type,
        "unit": unit,
    }
    
    for app_col, db_col in APP_TO_DB.items():
        if app_col in row_app:
            value = row_app[app_col]
            
            # Convert pandas NA/NaN to None
            if pd.isna(value):
                value = None
            # Convert empty strings to None for cleaner data
            elif isinstance(value, str) and value.strip() == "":
                value = None
            
            payload[db_col] = value
    
    # Upsert with conflict on (fleet_type, unit)
    sb.table(TABLE_NAME).upsert(
        payload,
        on_conflict="fleet_type,unit"
    ).execute()


def delete_vehicle_db(fleet_type: str, unit: str) -> None:
    """
    Delete a vehicle from Supabase.
    
    Args:
        fleet_type: "OP" or "UG"
        unit: the vehicle unit identifier
    """
    sb = get_supabase_client()
    
    sb.table(TABLE_NAME).delete().eq(
        "fleet_type", fleet_type
    ).eq(
        "unit", str(unit).strip()
    ).execute()
