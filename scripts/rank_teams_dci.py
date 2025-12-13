#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility — Team Ranking Report 
==============================================

Generates a Defensive Coverage Index (DCI) ranking report for NFL teams.
Inputs:
    1. metrics_playlevel_supervised.parquet 
    2. supplementary_data.csv 
"""

import pandas as pd
import os

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS 
# ==========================================

# Ruta a tu archivo de métricas (DCI Supervisado)
METRICS_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_supervised.parquet"

# Ruta a los datos crudos originales 
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")

# ==========================================
# 2. CARGA DE DATOS
# ==========================================
print("[INFO] Cargando métricas y datos desde Lustre...", flush=True)

try:
    df_metrics = pd.read_parquet(METRICS_PATH)
    print(f"   -> Métricas cargadas: {len(df_metrics)} jugadas.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el archivo de métricas en: {METRICS_PATH}")
    print(f"        Error: {e}")
    exit()

try:
    df_supp = pd.read_csv(SUPP_PATH, low_memory=False)
    print(f"   -> Datos suplementarios cargados.")
except Exception as e:
    print(f"[ERROR] No se encontró supplementary_data.csv en: {SUPP_PATH}")
    print(f"        Error: {e}")
    exit()

# ==========================================
# 3. LIMPIEZA Y UNIÓN
# ==========================================
print("[INFO] Procesando tablas...", flush=True)

# Estandarizar nombres a snake_case
rename_map = {
    'gameId': 'game_id',
    'playId': 'play_id',
    'defensiveTeam': 'defensive_team',
    'possessionTeam': 'possession_team',
    'week': 'week',
    'homeTeamAbbr': 'home_team',
    'visitorTeamAbbr': 'away_team'
}
df_supp.rename(columns=rename_map, inplace=True)

# Selección de columnas útiles
cols_to_use = ['game_id', 'play_id', 'defensive_team', 'possession_team']
if 'week' in df_supp.columns: cols_to_use.append('week')
if 'home_team' in df_supp.columns: cols_to_use.append('home_team')
if 'away_team' in df_supp.columns: cols_to_use.append('away_team')

df_supp_clean = df_supp[cols_to_use].copy()

# UNIÓN (Merge)
merged_df = df_metrics.merge(df_supp_clean, on=['game_id', 'play_id'], how='inner')
print(f"✅ Total de jugadas unidas y listas para análisis: {len(merged_df)}")

# ==========================================
# 4. CÁLCULO DE RANKINGS
# ==========================================

# --- Lógica para determinar el Rival ---
def get_opponent(row):
    if 'home_team' in row and pd.notna(row['home_team']):
        if row['defensive_team'] == row['home_team']:
            return row['away_team']
        return row['home_team']
    return row['possession_team']

# --- Agrupación ---
group_cols = ['game_id', 'defensive_team']
if 'week' in merged_df.columns:
    group_cols.insert(1, 'week')

# Promedio de DCI
team_stats = merged_df.groupby(group_cols)['dci_supervised'].mean().reset_index()

# Recuperar metadatos (Rival)
# Obtenemos la info del primer registro de cada grupo para no perder el contexto
meta_cols = group_cols + ['home_team', 'away_team', 'possession_team']
available_meta = [c for c in meta_cols if c in merged_df.columns]

# Truco para recuperar datos categóricos post-groupby
info_map = merged_df.groupby(group_cols)[available_meta].first().reset_index(drop=True)

# Combinamos los promedios con la info del partido
# (Al usar groupby igual, los índices deberían alinearse, pero hacemos merge por seguridad si los índices cambiaron)
# Simplemente reasignamos columnas si el orden se mantuvo, o hacemos merge.
# Haremos merge con las columnas llave.
team_stats = team_stats.merge(
    merged_df[available_meta].drop_duplicates(subset=group_cols),
    on=group_cols,
    how='left'
)

team_stats['opponent'] = team_stats.apply(get_opponent, axis=1)

# Limpieza final del reporte
if 'week' in team_stats.columns:
    final_ranking = team_stats[['week', 'defensive_team', 'opponent', 'dci_supervised']]
    final_ranking.columns = ['Semana', 'Equipo_Defensivo', 'Rival', 'DCI_Promedio']
else:
    final_ranking = team_stats[['defensive_team', 'opponent', 'dci_supervised']]
    final_ranking.columns = ['Equipo_Defensivo', 'Rival', 'DCI_Promedio']

# Ordenar (Mayor DCI es mejor)
final_ranking = final_ranking.sort_values(by='DCI_Promedio', ascending=False).reset_index(drop=True)

# ==========================================
# 5. REPORTE EN CONSOLA
# ==========================================

print("\n" + "="*60)
print("TOP 10 MEJORES ACTUACIONES DEFENSIVAS (POR PARTIDO)")
print("="*60)
print(final_ranking.head(10))

print("\n" + "="*60)
print("TOP 10 PEORES ACTUACIONES DEFENSIVAS (POR PARTIDO)")
print("="*60)
print(final_ranking.tail(10))

# Ranking General de Temporada
print("\n" + "="*60)
print("RANKING GENERAL DE LA TEMPORADA (PROMEDIO GLOBAL)")
print("="*60)
season_ranking = final_ranking.groupby('Equipo_Defensivo')['DCI_Promedio'].mean().reset_index()
season_ranking = season_ranking.sort_values(by='DCI_Promedio', ascending=False).reset_index(drop=True)
print(season_ranking.head(10))

print("\n[INFO] Ejecución finalizada.")

'''

# ==========================================
# 5.b. NORMALIZACIÓN (Hacerlo legible)
# ==========================================
# Vamos a convertir el DCI crudo en un "DCI Score" de 0 a 100
# Usamos Min-Max Scaling basado en los límites observados en tus datos

min_val = final_ranking['DCI_Promedio'].min()
max_val = final_ranking['DCI_Promedio'].max()

def normalize_score(val):
    # Fórmula: (Valor - Min) / (Max - Min) * 100
    return ((val - min_val) / (max_val - min_val)) * 100

final_ranking['DCI_Score_Adjusted'] = final_ranking['DCI_Promedio'].apply(normalize_score)

# Reordenamos columnas
final_ranking = final_ranking[['Semana', 'Equipo_Defensivo', 'Rival', 'DCI_Promedio', 'DCI_Score_Adjusted']]

# Ordenamos por el nuevo Score
final_ranking = final_ranking.sort_values(by='DCI_Score_Adjusted', ascending=False).reset_index(drop=True)

print("\n" + "="*60)
print("TOP 10 MEJORES ACTUACIONES (ESCALA 0-100)")
print("="*60)
# Formateamos para que se vea bonito
print(final_ranking.head(10).to_string(formatters={'DCI_Promedio': '{:,.4f}'.format, 'DCI_Score_Adjusted': '{:,.1f}'.format}))

'''