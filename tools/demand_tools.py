# tools/demand_tools.py
from __future__ import annotations
import pandas as pd
from typing import Optional
import numpy as np

def _delivered_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Return minimal orders table filtered to 'delivered' status with timestamps."""
    if "order_status" not in orders.columns or "order_purchase_timestamp" not in orders.columns:
        raise ValueError("orders DataFrame must contain 'order_status' and 'order_purchase_timestamp' columns.")
    return orders.loc[orders["order_status"] == "delivered", ["order_id", "order_purchase_timestamp"]].copy()


def _product_category_map(products: pd.DataFrame, category_translation: pd.DataFrame) -> pd.DataFrame:
    """
    Build a mapping table: product_id -> product_category (english).
    Returns DataFrame with columns ['product_id', 'product_category'].
    """
    if "product_id" not in products.columns or "product_category_name" not in products.columns:
        # If product table has different column names adjust or raise for clarity
        raise ValueError("products must contain 'product_id' and 'product_category_name' columns.")
    # merge with translation if available
    if category_translation is not None and "product_category_name" in category_translation.columns:
        merged = products[["product_id", "product_category_name"]].merge(
            category_translation,
            on="product_category_name",
            how="left"
        )
        # prefer the english translation when available
        merged["product_category"] = merged.get("product_category_name_english", merged["product_category_name"]).fillna("unknown")
        return merged[["product_id", "product_category"]]
    else:
        products = products.copy()
        products["product_category"] = products["product_category_name"].fillna("unknown")
        return products[["product_id", "product_category"]]


def get_daily_demand(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: Optional[pd.DataFrame] = None,
    product_id: Optional[str] = None,
    product_category: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fill_missing_dates: bool = False,
    min_date_range_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return daily units-sold time series (product_id x date) with product_category.

    Parameters
    ----------
    orders : pd.DataFrame
        orders.csv loaded with at least 'order_id', 'order_purchase_timestamp', 'order_status'
    order_items : pd.DataFrame
        order_items.csv with at least 'order_id', 'product_id', 'order_item_id' (one row per unit)
    products : pd.DataFrame
        products.csv with 'product_id' and 'product_category_name'
    category_translation : Optional[pd.DataFrame]
        translation csv mapping Portuguese category -> English. If None, will use product_category_name as-is.
    product_id : Optional[str]
        if provided, filter to this product_id (exact match)
    product_category : Optional[str]
        if provided, filter to this product_category (English name; applied after translation)
    start : Optional[str]
        inclusive start date in 'YYYY-MM-DD' format. If None, no lower bound.
    end : Optional[str]
        inclusive end date in 'YYYY-MM-DD' format. If None, no upper bound.
    fill_missing_dates : bool
        if True, will reindex each product_id to include all dates in [start,end] and fill units_sold with 0.
        Requires start and end (or infers from data if both are None).
    min_date_range_days : Optional[int]
        if provided, ensures the returned date range covers at least this many days (expand with zeros either side).
        Useful to make the series length consistent for modeling.

    Returns
    -------
    pd.DataFrame
        columns: ['product_id', 'product_category', 'date' (datetime64[ns]), 'units_sold' (int)]
        Sorted by ['product_id','date'].
    """

    # 1) Validate inputs quickly
    if orders is None or order_items is None or products is None:
        raise ValueError("orders, order_items and products DataFrames are required.")

    # 2) Filter delivered orders and join with order_items
    delivered = _delivered_orders(orders)  # contains order_id, order_purchase_timestamp
    df = delivered.merge(order_items, on="order_id", how="inner")
    # Expect order_items to have 'product_id' and possibly 'order_item_id'
    if "product_id" not in df.columns:
        raise ValueError("order_items must contain 'product_id' column.")

    # 3) Attach product_category (english preferred)
    cat_map = _product_category_map(products, category_translation)
    df = df.merge(cat_map, on="product_id", how="left")

    # 4) Normalize date column and filters
    df["date"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.date
    df["date"] = pd.to_datetime(df["date"])  # make it datetime64[ns] with midnight

    if product_id is not None:
        df = df[df["product_id"] == product_id]

    if product_category is not None:
        df = df[df["product_category"] == product_category]

    if start is not None:
        start_ts = pd.to_datetime(start)
        df = df[df["date"] >= start_ts]

    if end is not None:
        end_ts = pd.to_datetime(end)
        df = df[df["date"] <= end_ts]

    # 5) Aggregate: count rows per product_id x date (each order_item row = 1 unit)
    daily = (
        df.groupby(["product_id", "product_category", "date"])
          .size()
          .reset_index(name="units_sold")
    )

    # 6) Optionally fill missing dates (for time series modeling)
    if fill_missing_dates:
        # derive range
        if start is None:
            overall_start = daily["date"].min() if not daily.empty else pd.to_datetime("1970-01-01")
        else:
            overall_start = pd.to_datetime(start)
        if end is None:
            overall_end = daily["date"].max() if not daily.empty else pd.to_datetime("1970-01-01")
        else:
            overall_end = pd.to_datetime(end)

        # optionally ensure min_date_range_days
        if min_date_range_days is not None:
            current_len = (overall_end - overall_start).days + 1
            if current_len < min_date_range_days:
                # expand range equally on both sides when possible
                extra = min_date_range_days - current_len
                left_extra = extra // 2
                right_extra = extra - left_extra
                overall_start = overall_start - pd.Timedelta(days=left_extra)
                overall_end = overall_end + pd.Timedelta(days=right_extra)

        all_dates = pd.date_range(start=overall_start, end=overall_end, freq="D")
        # cartesian product of product_ids and all_dates
        product_ids = daily["product_id"].unique().tolist()
        if product_id is not None:
            product_ids = [product_id]  # keep in given order

        # build base frame and merge
        base = pd.MultiIndex.from_product([product_ids, all_dates], names=["product_id", "date"]).to_frame(index=False)
        # Need product_category mapping per product_id
        pcats = daily[["product_id", "product_category"]].drop_duplicates().set_index("product_id")
        base = base.merge(pcats.reset_index(), on="product_id", how="left")
        # join daily counts
        daily = base.merge(daily, on=["product_id", "product_category", "date"], how="left")
        daily["units_sold"] = daily["units_sold"].fillna(0).astype(int)

    # 7) Final housekeeping: sort and types
    if "units_sold" in daily.columns:
        daily["units_sold"] = daily["units_sold"].astype(int)

    daily = daily.sort_values(["product_id", "date"]).reset_index(drop=True)

    return daily




def compute_product_unpredictability(
    daily_demand_df: pd.DataFrame,
    min_active_days: int = 14,
    min_total_units: int = 20,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Computes demand variability per product using daily demand.

    It returns a unitless
    'unpredictability_score' = cv * log1p(total_units_sold).

    Parameters
    ----------
    daily_demand_df : pd.DataFrame
        Output of get_daily_demand(). Must contain:
        ['product_id', 'product_category', 'date', 'units_sold']
    min_active_days : int
        Filter out products with too few selling days.
    min_total_units : int
        Filter out products with too few total units sold.
    top_n : int
        Return only the top N products by unpredictability_score.

    Returns
    -------
    pd.DataFrame with one row per (product_id, product_category) containing:
    mean_units_sold, std_units_sold, total_units_sold, days_active, cv,
    unpredictability_score
    """

    required_cols = {"product_id", "product_category", "date", "units_sold"}
    missing = required_cols - set(daily_demand_df.columns)
    if missing:
        raise ValueError(f"daily_demand_df missing columns: {missing}")

    df = daily_demand_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    stats = (
        df.groupby(["product_id", "product_category"])["units_sold"]
          .agg(
              mean_units_sold="mean",
              std_units_sold="std",
              total_units_sold="sum",
              days_active="count",
              first_date=("min"),
              last_date=("max"),
          )
          .reset_index()
    )

    # std is NaN when days_active == 1
    stats["std_units_sold"] = stats["std_units_sold"].fillna(0.0)

    # CV = std / mean (guard divide-by-zero)
    stats["cv"] = np.where(
        stats["mean_units_sold"] > 0,
        stats["std_units_sold"] / stats["mean_units_sold"],
        np.nan
    )

    # Filter sparse products
    stats = stats[
        (stats["days_active"] >= min_active_days) &
        (stats["total_units_sold"] >= min_total_units)
    ].copy()

    # Unitless risk index: volatility × (compressed) volume
    stats["unpredictability_score"] = stats["cv"] * np.log1p(stats["total_units_sold"])

    stats = stats.sort_values(
        ["unpredictability_score", "total_units_sold"],
        ascending=[False, False]
    )

    return stats.head(top_n).reset_index(drop=True)


def classify_demand_types(
    daily_demand_df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    min_total_days: int = 60,
    min_nonzero_days: int = 5,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Classifies each product into Smooth / Erratic / Intermittent / Lumpy using ADI and CV^2.

    daily_demand_df expected columns:
      ['product_id', 'product_category', 'date', 'units_sold']

    Notes:
    - This works best when daily_demand_df has missing dates filled with 0 for each product.
    """

    required_cols = {"product_id", "product_category", "date", "units_sold"}
    missing = required_cols - set(daily_demand_df.columns)
    if missing:
        raise ValueError(f"daily_demand_df missing columns: {missing}")

    df = daily_demand_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]

    # total days observed per product
    total_days = df.groupby(["product_id", "product_category"])["date"].nunique().rename("total_days")

    # non-zero days + stats on non-zero demand sizes
    nonzero = df[df["units_sold"] > 0].copy()

    nonzero_days = nonzero.groupby(["product_id", "product_category"])["date"].nunique().rename("nonzero_days")
    nonzero_mean = nonzero.groupby(["product_id", "product_category"])["units_sold"].mean().rename("nonzero_mean")
    nonzero_std = nonzero.groupby(["product_id", "product_category"])["units_sold"].std().fillna(0.0).rename("nonzero_std")

    out = (
        pd.concat([total_days, nonzero_days, nonzero_mean, nonzero_std], axis=1)
          .reset_index()
    )

    # Fill missing (products with zero nonzero days in window)
    out["nonzero_days"] = out["nonzero_days"].fillna(0).astype(int)
    out["nonzero_mean"] = out["nonzero_mean"].fillna(0.0)
    out["nonzero_std"] = out["nonzero_std"].fillna(0.0)

    # Guardrails: require enough observation
    out = out[out["total_days"] >= min_total_days].copy()

    # ADI
    out["adi"] = np.where(out["nonzero_days"] > 0, out["total_days"] / out["nonzero_days"], np.inf)

    # CV^2 (on non-zero demand)
    nonzero_cv = np.where(out["nonzero_mean"] > 0, out["nonzero_std"] / out["nonzero_mean"], np.nan)
    out["cv2"] = nonzero_cv ** 2

    # Optional: filter products with too few non-zero days (unstable stats)
    out = out[out["nonzero_days"] >= min_nonzero_days].copy()

    # Classification thresholds
    ADI_T = 1.32
    CV2_T = 0.49

    def _classify(row) -> str:
        adi = row["adi"]
        cv2 = row["cv2"]
        if np.isinf(adi) or np.isnan(cv2):
            return "insufficient_data"
        if adi < ADI_T and cv2 < CV2_T:
            return "smooth"
        if adi < ADI_T and cv2 >= CV2_T:
            return "erratic"
        if adi >= ADI_T and cv2 < CV2_T:
            return "intermittent"
        return "lumpy"

    out["demand_type"] = out.apply(_classify, axis=1)

    # Helpful extra columns for later decisions
    out["intermittency_rate"] = np.where(out["total_days"] > 0, out["nonzero_days"] / out["total_days"], 0.0)

    # Sort: most intermittent first (high ADI), then most variable (high cv2)
    out = out.sort_values(["adi", "cv2"], ascending=[False, False]).reset_index(drop=True)

    if top_n is not None:
        out = out.head(top_n)

    return out
