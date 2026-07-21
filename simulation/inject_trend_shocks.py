import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

def inject_trend_effects(df):
    """
    [SIMULATION ONLY]
    Wskrzykuje nielioniowy wplyw trendu na marki, wykorzystujac asymetryczna
    rekcje na spadki i wzrosty
    """

    df['simulated_units_sold'] = df['units_sold']
    momentum = df['trend_momentum']

    # domyslenie dla wartosci momentum z przedzialu [-0.5, 0.5] nie przyznajemy "bonusu"
    trend_impact = np.zeros(df.shape[0])

    # poztywyny, silny wplyw trendu - efekt FOMO
    positive_mask = momentum > 0.5
    trend_impact[positive_mask] = (momentum[positive_mask] - 0.5) * 0.25

    #negatywny, slabszy wplyw trendu - efekt zapomnienia
    negative_mask = momentum < -0.5
    trend_impact[negative_mask] = (momentum[negative_mask] + 0.5) * 0.1

    #limity wplywu trendu (-30% i +50%)
    trend_impact = np.clip(trend_impact, -0.3, 0.5)

    multiplier = 1 + trend_impact
    df['simulated_units_sold'] = df['simulated_units_sold'] * multiplier

    # dodanie szumu, aby utrudnic prace modelom
    noise = np.random.normal(0, 1, df.shape[0])
    df['simulated_units_sold'] = np.round(df['simulated_units_sold'] + noise)

    # zabezpieczenie przed ujemnymi wartosciami
    df['simulated_units_sold'] = np.where(df['simulated_units_sold'] > 0, df['simulated_units_sold'], 0)

    # usuwanie starej kolumny w celu zabezpieczenia przed data-leakage
    df = df.drop(columns=['units_sold'])
    df = df.rename(columns={'simulated_units_sold': 'units_sold'})

    return df

def main():
    input_path = PROCESSED_DATA_DIR / "master_dataset.parquet"
    output_path = PROCESSED_DATA_DIR / "master_dataset_simulated.parquet"

    df_master = pd.read_parquet(input_path)
    df_simulated = inject_trend_effects(df_master)

    df_simulated.to_parquet(output_path)

if __name__ == "__main__":
    main()
