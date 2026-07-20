import time
import random
import logging
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq

# 1. Konfiguracja Loggera
logging.basicConfig(
    level=logging.INFO, format="{asctime} - {levelname} - {message}", style="{"
)
logger = logging.getLogger(__name__)

# 2. Definicja ścieżek (dynamicznie względem pliku skryptu)
# Zakładamy strukturę: market-pulse-dashboard/data_pipeline/01_fetch_trends.py
# Chcemy trafić do: market-pulse-dashboard/data/raw/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_DIR = RAW_DATA_DIR / "checkpoints"

# Upewniamy się, że folder docelowy istnieje
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REGION_TZ_MAP = {
    "AU": -600,  # Sydney
    "DE": -60,  # Berlin
    "US": 300,  # Nowy Jork
    "UK": 0,  # Londyn
    "CA": 300,  # Toronto
    "IN": -330,  # New Delhi
}


def fetch_batch_for_region(brand_name, region, timeframe="today 3-m"):
    """Funkcja pobierająca paczkę danych i obsługująca połączenie z API."""

    local_tz = REGION_TZ_MAP.get(region, 0)
    pytrends = TrendReq(hl="pl-PL", tz=local_tz, retries=2, backoff_factor=10)

    # Tłumaczymy kod UK na GB wymagany przez API Google Trends
    api_region = "GB" if region == "UK" else region

    logger.info(f"Budowanie zapytania dla regionu {region}, marki: {brand_name}")

    try:
        pytrends.build_payload(
            kw_list=[brand_name], timeframe=timeframe, geo=api_region
        )
        df = pytrends.interest_over_time()

        if df.empty:
            logger.warning(f"Brak danych dla zapytania: {brand_name} w {region}")
            return None

        # Sprzątanie technicznej kolumny
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        # Zmieniamy strukture wyplutego df'a w celu łatwiejszej, pozniejszsej agregacji
        df = df.rename(columns={brand_name: "trend_index"})
        df["region"] = region
        df["brand"] = brand_name
        df = df.reset_index()
        return df[["date", "region", "brand", "trend_index"]]

    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych z Google Trends: {e}")
        return None


def main():
    # Pełny okres z jakiego pochadza zamowienia
    df_source = pd.read_csv(RAW_DATA_DIR / "retail_pricing_demand_100k.csv")
    df_source["date"] = pd.to_datetime(df_source["date"])
    start_date = df_source["date"].min().strftime("%Y-%m-%d")
    end_date = df_source["date"].max().strftime("%Y-%m-%d")
    timeframe = f"{start_date} {end_date}"

    # nazwy obecnych kategorii wejściowych
    regions = df_source.region.unique().tolist()
    brand_namees = df_source.brand.unique().tolist()

    logger.info("Rozpoczęcie pobierania danych...")

    for region in regions:
        for brand in brand_namees:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{region}_{brand}.parquet"
            if checkpoint_path.exists():
                logger.info(f"Pominięto (już pobrano): {brand} w {region}")
                continue

            logger.info(f"Pobieranie: {brand} ({region})")
            df_trend = fetch_batch_for_region(brand, region, timeframe)

            if df_trend is not None:
                df_trend.to_parquet(checkpoint_path, index=False)
                logger.info(f"Zapisano checkpoint dla {brand} ({region})")

            sleep_time = random.randint(60, 120)
            logger.info(f"Uśpienie potoku na {sleep_time} sekund...")
            time.sleep(sleep_time)

    # Łączenie i zapis wyników
    checkpoint_files = list(CHECKPOINT_DIR.glob("checkpoint_*.parquet"))

    if not checkpoint_files:
        logger.warning("Brak checkpointów do zlaczenia")
        return

    all_dfs = [pd.read_parquet(f) for f in checkpoint_files]
    final_df = pd.concat(all_dfs, ignore_index=True)

    output_path = RAW_DATA_DIR / "google_trends.parquet"
    final_df.to_parquet(output_path, index=False)

    logger.info(f" Sukces! Scalono {len(checkpoint_files)} plików do: {output_path}")


if __name__ == "__main__":
    main()
