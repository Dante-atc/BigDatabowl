import pandas as pd
import numpy as np

# Paths
METRICS_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPP_PATH = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

df = pd.read_parquet(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Normalize and Merge
supp.rename(columns={"gameId": "game_id", "playId": "play_id", "passResult": "pass_result"}, inplace=True)
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")

# STRICT FILTER: Clear Passes Only
merged = merged[merged['pass_result'].isin(['C', 'I', 'IN', 'S'])]

print(f"Analyzing {len(merged)} pure passes...")

# --- THE LITMUS TEST ---
# Group by result.
# HYPOTHESIS:
# - Complete Passes (C) should have HIGHER distance to ideal (worse coverage).
# - Incomplete Passes (I) should have LOWER distance to ideal (better coverage).

group = merged.groupby('pass_result')[['distance_to_ideal', 'spacing_proxy', 'integrity_proxy']].mean()
print("\n--- AVERAGES BY RESULT ---")
print(group)

# Calculate the "Gap" (Difference)
gap = group.loc['C', 'distance_to_ideal'] - group.loc['I', 'distance_to_ideal']
print(f"\nGAP (Complete - Incomplete): {gap:.4f}")

if gap > 0:
    print(" SIGNAL DETECTED! Bad coverages (distant) allow more complete passes.")
    print("    The optimizer will be able to exploit this.")
else:
    print(" RED ALERT: 'Good' coverages (close) are allowing passes.")
    print("    We need to rethink what 'Ideal' means.")
