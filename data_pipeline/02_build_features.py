import pandas as pd
from pathlib import Path
import logging

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_zscore(df):
    """Wylicza wewnatrzgrupowy zscore"""

    groups = df.groupby(["region", "brand"])["trend_index"]

    means = groups.transform("mean")
    stds = groups.transform("std").replace(0, 1)

    df["zscored_trend_index"] = (df["trend_index"] - means) / stds
    return df


def calculate_rolling_features(df):
    """Wylicza wewnatrzgrupowe srednie kroczace: MA, MA7 i ich delte"""

    df = df.sort_values(by=["region", "brand", "date"]).reset_index(drop=True)

    groups = df.groupby(["region", "brand"])["zscored_trend_index"]

    df["ma_3"] = groups.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["ma_7"] = groups.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())

    df[["ma_3", "ma_7"]] = df[["ma_3", "ma_7"]].fillna(0)

    df["trend_momentum"] = df["ma_3"] - df["ma_7"]

    final_cols = ["date", "region", "brand", "trend_momentum"]
    return df[final_cols]


def main():
    """Wykonuje logike i zapisuje na dysk"""

    df = pd.read_parquet(RAW_DATA_DIR / "google_trends.parquet")
    df = calculate_zscore(df)
    df = calculate_rolling_features(df)

    df.to_parquet(PROCESSED_DATA_DIR / "trend_features.parquet")


if __name__ == "__main__":
    main()
