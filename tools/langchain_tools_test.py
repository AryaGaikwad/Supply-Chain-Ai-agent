# tools/langchain_tools.py
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any, Literal
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
        data_dir = os.getenv("OLIST_DATA_DIR", "./data")
        _DATA = load_olist_data(data_dir)
    return _DATA


def _df_to_records(df: pd.DataFrame, max_rows: int = 50) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.head(max_rows).copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict(orient="records")


# -------------------------
# 1) Data catalog tool
# -------------------------
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

@tool
def get_data_dictionary() -> Dict[str, Any]:
    """
    Return a dictionary of available datasets, key columns, and recommended joins.

    This helps the agent decide what can be computed and how to join tables safely.
    Use this tool before answering questions like:
    - "top products by demand"
    - "top categories by demand"
    - "demand for product_id X"
    """
    return {
        "tables": {
            "orders": {
                "purpose": "One row per order with status + timestamps",
                "key_columns": {
                    "order_id": "unique order id",
                    "order_status": "e.g., delivered/canceled",
                    "order_purchase_timestamp": "when the order was placed",
                    "order_delivered_customer_date": "when the order was delivered to customer"
                },
                "notes": [
                    "For demand, usually filter order_status == 'delivered'.",
                    "Use order_purchase_timestamp for date filters."
                ]
            },
            "order_items": {
                "purpose": "One row per item line in an order (each row counts as 1 unit sold)",
                "key_columns": {
                    "order_id": "joins to orders.order_id",
                    "product_id": "joins to products.product_id",
                    "order_item_id": "sequential item number within the order (NOT quantity)",
                    "price": "item price",
                    "shipping_charges": "shipping charge"
                },
                "notes": [
                    "Demand (units) is computed by counting rows (or sum of 1 per row)."
                ]
            },
            "products": {
                "purpose": "Product metadata",
                "key_columns": {
                    "product_id": "joins to order_items.product_id",
                    "product_category_name": "category name (Portuguese)"
                }
            },
            "category_translation": {
                "purpose": "Translate category to English",
                "key_columns": {
                    "product_category_name": "joins to products.product_category_name",
                    "product_category_name_english": "English category"
                }
            }
        },
        "recommended_joins": [
            {
                "step": 1,
                "join": "orders INNER JOIN order_items ON order_id",
                "why": "connect order status/timestamp to items sold"
            },
            {
                "step": 2,
                "join": "… LEFT JOIN products ON product_id",
                "why": "attach category to items"
            },
            {
                "step": 3,
                "join": "… LEFT JOIN category_translation ON product_category_name",
                "why": "get English category when available"
            }
        ],
        "common_metrics": {
            "units_sold": "COUNT(order_items rows) after filtering delivered orders",
            "demand_by_product": "GROUP BY product_id; sum units_sold",
            "demand_by_category": "GROUP BY product_category (English); sum units_sold",
        }
    }


@tool
def top_demand_products(
    top_n: int = 3,
    delivered_only: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_category: bool = True
) -> List[Dict[str, Any]]:
    """
    Return top-N products by total demand (units sold).

    Demand definition:
    - 1 row in order_items = 1 unit sold.
    - Optionally filter to delivered orders only.

    Parameters:
    - top_n: how many products to return
    - delivered_only: if True, only count orders with order_status == 'delivered'
    - start/end: optional YYYY-MM-DD date window (uses orders.order_purchase_timestamp)
    - include_category: if True, include English category when available
    """
    data = _get_data()

    orders = data.orders.copy()
    if delivered_only and "order_status" in orders.columns:
        orders = orders[orders["order_status"] == "delivered"].copy()

    # time filter uses purchase timestamp
    if start is not None:
        orders = orders[orders["order_purchase_timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        orders = orders[orders["order_purchase_timestamp"] <= pd.to_datetime(end)]

    # join orders -> order_items
    df = orders[["order_id", "order_purchase_timestamp"]].merge(
        data.order_items[["order_id", "product_id"]],
        on="order_id",
        how="inner"
    )

    if include_category:
        # products + translation => English category
        prod = data.products[["product_id", "product_category_name"]].copy()
        df = df.merge(prod, on="product_id", how="left")
        df = df.merge(
            data.category_translation,
            on="product_category_name",
            how="left"
        )
        df["product_category"] = df["product_category_name_english"].fillna(df["product_category_name"]).fillna("unknown")

        ranked = (
            df.groupby(["product_id", "product_category"])
              .size()
              .reset_index(name="total_units_sold")
              .sort_values("total_units_sold", ascending=False)
        )
    else:
        ranked = (
            df.groupby(["product_id"])
              .size()
              .reset_index(name="total_units_sold")
              .sort_values("total_units_sold", ascending=False)
        )

    return _df_to_records(ranked, max_rows=int(top_n))


@tool
def demand_by_category(
    top_n: int = 10,
    delivered_only: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return top categories by demand (units sold).

    Uses the same demand definition:
    - 1 row in order_items = 1 unit sold.
    - Optionally filters to delivered orders only.
    - Category is English when possible (via translation table).
    """
    data = _get_data()

    orders = data.orders.copy()
    if delivered_only and "order_status" in orders.columns:
        orders = orders[orders["order_status"] == "delivered"].copy()

    if start is not None:
        orders = orders[orders["order_purchase_timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        orders = orders[orders["order_purchase_timestamp"] <= pd.to_datetime(end)]

    df = orders[["order_id"]].merge(
        data.order_items[["order_id", "product_id"]],
        on="order_id",
        how="inner"
    )

    df = df.merge(
        data.products[["product_id", "product_category_name"]],
        on="product_id",
        how="left"
    ).merge(
        data.category_translation,
        on="product_category_name",
        how="left"
    )

    df["product_category"] = df["product_category_name_english"].fillna(df["product_category_name"]).fillna("unknown")

    ranked = (
        df.groupby("product_category")
          .size()
          .reset_index(name="total_units_sold")
          .sort_values("total_units_sold", ascending=False)
    )

    return _df_to_records(ranked, max_rows=int(top_n))


@tool
def product_demand_summary(
    product_id: str,
    delivered_only: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a simple demand summary for a single product_id.

    Output includes:
    - total_units_sold (count of order_items rows)
    - active_days (unique purchase dates)
    - first_date / last_date (based on purchase timestamp)
    - product_category (English when available)
    """
    data = _get_data()

    orders = data.orders.copy()
    if delivered_only and "order_status" in orders.columns:
        orders = orders[orders["order_status"] == "delivered"].copy()

    if start is not None:
        orders = orders[orders["order_purchase_timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        orders = orders[orders["order_purchase_timestamp"] <= pd.to_datetime(end)]

    df = orders[["order_id", "order_purchase_timestamp"]].merge(
        data.order_items[["order_id", "product_id"]],
        on="order_id",
        how="inner"
    )
    df = df[df["product_id"] == product_id].copy()

    if df.empty:
        return {"product_id": product_id, "found": False}

    # category
    cat = data.products[["product_id", "product_category_name"]].merge(
        data.category_translation,
        on="product_category_name",
        how="left"
    )
    cat["product_category"] = cat["product_category_name_english"].fillna(cat["product_category_name"]).fillna("unknown")
    product_category = cat.loc[cat["product_id"] == product_id, "product_category"]
    category_val = str(product_category.iloc[0]) if not product_category.empty else "unknown"

    # demand summary
    df["date"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.date
    total_units = int(len(df))
    active_days = int(df["date"].nunique())
    first_date = str(pd.to_datetime(df["order_purchase_timestamp"]).min().date())
    last_date = str(pd.to_datetime(df["order_purchase_timestamp"]).max().date())

    return {
        "product_id": product_id,
        "product_category": category_val,
        "found": True,
        "total_units_sold": total_units,
        "active_days": active_days,
        "first_date": first_date,
        "last_date": last_date,
    }


TOOLS = [
    get_data_dictionary,
    top_demand_products,
    demand_by_category,
    product_demand_summary,
    get_daily_demand,
    compute_product_unpredictability,
    classify_demand_types,
]
