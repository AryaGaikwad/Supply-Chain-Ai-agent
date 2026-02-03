# tools/langchain_tools.py
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
import pandas as pd

from langchain_core.tools import tool

from tools.data_loader import load_olist_data
from tools.demand_tools import (
    get_daily_demand as _get_daily_demand_fn,
    compute_product_unpredictability as _compute_product_unpredictability_fn,
    classify_demand_types as _classify_demand_types_fn,
)

_DATA = None

def _get_data():
    global _DATA
    if _DATA is None:
        data_dir = os.getenv("OLIST_DATA_DIR", "./olist_data") 
        _DATA = load_olist_data(data_dir)
    return _DATA

def _df_to_records(df: pd.DataFrame, max_rows: int = 50) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.head(max_rows).copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="ignore")
        if pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")

@tool
def get_daily_demand(
    product_id: Optional[str] = None,
    product_category: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fill_missing_dates: bool = False,
    max_rows: int = 50,
) -> List[Dict[str, Any]]:
    """Get daily units/demand sold per (product_id, date) with product_category. Supports filters and optional zero-filling."""
    data = _get_data()
    df = _get_daily_demand_fn(
        data.orders, data.order_items, data.products, data.category_translation,
        product_id=product_id, product_category=product_category,
        start=start, end=end, fill_missing_dates=fill_missing_dates,
    )
    return _df_to_records(df, max_rows=max_rows)


@tool
def compute_product_unpredictability(top_n: int = 20) -> List[Dict[str, Any]]:
    """Return top-N unpredictable SKUs using score = CV * log1p(total_units_sold)."""
    data = _get_data()
    daily = _get_daily_demand_fn(data.orders, data.order_items, data.products, data.category_translation)
    df = _compute_product_unpredictability_fn(daily, top_n=int(top_n))
    cols = [
        "product_id", "product_category",
        "mean_units_sold", "std_units_sold", "cv",
        "total_units_sold", "days_active", "unpredictability_score",
    ]
    return _df_to_records(df[cols].copy(), max_rows=int(top_n))


@tool
def classify_demand_types(
    start: str = "2017-01-01",
    end: str = "2018-08-31",
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Classify SKUs into smooth/erratic/intermittent/lumpy using ADI and CV^2. Returns top-N rows."""
    data = _get_data()
    daily = _get_daily_demand_fn(
        data.orders, data.order_items, data.products, data.category_translation,
        start=start, end=end, fill_missing_dates=True
    )
    df = _classify_demand_types_fn(daily, min_total_days=60, min_nonzero_days=5, top_n=int(top_n))
    cols = ["product_id", "product_category", "total_days", "nonzero_days", "adi", "cv2", "demand_type"]
    return _df_to_records(df[cols].copy(), max_rows=int(top_n))

TOOLS = [get_daily_demand, compute_product_unpredictability, classify_demand_types]
