import pandas as pd
import numpy as np

# Rutas
METRICS_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPP_PATH = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

df = pd.read_parquet(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Normalizar y Merge
supp.rename(columns={"gameId": "game_id", "playId": "play_id", "passResult": "pass_result"}, inplace=True)
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")

# FILTRO ESTRICTO: Solo Pases Claros
merged = merged[merged['pass_result'].isin(['C', 'I', 'IN', 'S'])]

print(f"Analizando {len(merged)} pases puros...")

# --- LA PRUEBA DE FUEGO ---
# Agrupamos por resultado.
# HIPÓTESIS:
# - Pases Completos (C) deberían tener MAYOR distancia al ideal (peor cobertura).
# - Pases Incompletos (I) deberían tener MENOR distancia al ideal (mejor cobertura).

group = merged.groupby('pass_result')[['distance_to_ideal', 'spacing_proxy', 'integrity_proxy']].mean()
print("\n--- PROMEDIOS POR RESULTADO ---")
print(group)

# Calculamos el "Gap" (Diferencia)
gap = group.loc['C', 'distance_to_ideal'] - group.loc['I', 'distance_to_ideal']
print(f"\nGAP (Completo - Incompleto): {gap:.4f}")

if gap > 0:
    print("✅ ¡SEÑAL DETECTADA! Las coberturas malas (lejanas) permiten más pases completos.")
    print("   El optimizador podrá explotar esto.")
else:
    print("⚠️ ALERTA ROJA: Las coberturas 'buenas' (cercanas) están permitiendo pases.")
    print("   Necesitamos repensar qué significa 'Ideal'.")