import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


def marge_and_impute(df_sales, df_trends):
    """Łaczy zbudowane cechy z pierwotnym zbiorem i uzupełnia braki"""

    df_master = df_sales.merge(df_trends, on=["date", "region", "brand"], how="left")

    df_master["trend_momentum"] = df_master["trend_momentum"].fillna(0)

    return df_master


def main():
    """Funkcja orkierstujaca: czyta z dysku, wykonuje logike i zapisuje na dysk"""

    sales_path = RAW_DATA_DIR / "retail_pricing_demand_100k.csv"
    trends_path = PROCESSED_DATA_DIR / "trend_features.parquet"
    output_path = PROCESSED_DATA_DIR / "master_dataset.parquet"

    if not sales_path.exists() or not trends_path.exists():
        logger.error("Missing sales or trend files")
        return

    df_sales = pd.read_csv(sales_path, parse_dates=["date"])
    df_trends = pd.read_parquet(trends_path)

    df_trends["date"] = pd.to_datetime(df_trends["date"])

    df_master = marge_and_impute(df_sales, df_trends)
    df_master.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
