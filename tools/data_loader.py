# tools/data_loader.py

from dataclasses import dataclass
import pandas as pd
import os


@dataclass
class OlistData:
    """
    Container for all Olist tables.
    This is the single source of truth for raw data.
    """
    orders: pd.DataFrame
    order_items: pd.DataFrame
    products: pd.DataFrame
    category_translation: pd.DataFrame


def load_olist_data(data_dir: str) -> OlistData:
    """
    Loads Olist CSVs from disk and returns them as DataFrames.

    This function should be called ONCE at app startup.
    """

    orders = pd.read_csv(
        os.path.join(data_dir, "olist_orders_dataset.csv"),
        parse_dates=["order_purchase_timestamp"]
    )

    order_items = pd.read_csv(
        os.path.join(data_dir, "olist_order_items_dataset.csv")
    )

    products = pd.read_csv(
        os.path.join(data_dir, "olist_products_dataset.csv")
    )

    category_translation = pd.read_csv(
        os.path.join(data_dir, "product_category_name_translation.csv")
    )

    return OlistData(
        orders=orders,
        order_items=order_items,
        products=products,
        category_translation=category_translation,
    )
