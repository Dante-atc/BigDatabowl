import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Rutas
METRICS_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPP_PATH = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

print("--- CARGANDO DATOS ---")
df = pd.read_parquet(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Normalizar nombres en supp
rename_map = {"gameId": "game_id", "playId": "play_id", 
              "passResult": "pass_result", "expectedPointsAdded": "expected_points_added"}
supp.rename(columns=rename_map, inplace=True)

# Merge
print(f"Métricas filas: {len(df)}")
print(f"Supp filas: {len(supp)}")
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")
print(f"Merged filas: {len(merged)}")

print("\n--- CHEQUEO DE EPA (Ground Truth) ---")
print(merged["expected_points_added"].describe())
print(f"Nulos en EPA: {merged['expected_points_added'].isna().sum()}")
print(f"Ceros exactos en EPA: {(merged['expected_points_added'] == 0).sum()}")

print("\n--- CHEQUEO DE DISTANCIAS (Tu Métrica) ---")
print(merged["distance_to_ideal"].describe())

print("\n--- CORRELACIONES CRUDAS ---")
# Checamos correlación directa sin fórmula compleja
corrs = {}
targets = ["distance_to_ideal", "spacing_proxy", "integrity_proxy"]
for t in targets:
    # Limpiar Nulos
    clean = merged[[t, "expected_points_added"]].dropna()
    r, _ = pearsonr(clean[t], clean["expected_points_added"])
    corrs[t] = r
    print(f"{t} vs EPA: {r:.4f}")

print("\n--- MUESTRA DE DATOS ---")
print(merged[["game_id", "play_id", "distance_to_ideal", "expected_points_added", "pass_result"]].head(10))