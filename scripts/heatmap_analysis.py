#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 13: HEATMAP & CORRELATION ANALYSIS
========================================

SUMMARY
-------
This script generates comprehensive static visualizations from play-level 
animation data. It moves beyond frame-by-frame animation to analyze spatial 
distributions, stress correlations, and temporal trends across the entire play.

It produces high-resolution heatmaps and statistical plots that reveal:
1. Spatial dominance (Position Heatmaps).
2. Defensive integrity breakdowns (Stress Correlations).
3. Physical performance metrics (Movement & Speed Analysis).
4. Structural evolution of the play (Formation & Time Series Analysis).

METHODOLOGY
-----------
1. Metric Calculation: Computes derived metrics like speed magnitude, distance 
   from the "ball" (centroid), and categorizes field zones (e.g., Red Zone, Deep Own).
2. Spatial Binning: Uses 2D histograms to generate density heatmaps for 
   offensive and defensive positioning.
3. Statistical Correlation: Calculates Pearson and Spearman correlations between 
   stress (isolation) and physical metrics (speed, position).
4. Time-Series Aggregation: Aggregates frame-level data to visualize how 
   stress and formation width/depth evolve over the duration of the play.

INPUTS
------
1. animation_data_*.csv (Generated from Phase 12 or similar scripts)

OUTPUTS
-------
- analysis_output/position_heatmaps.png
- analysis_output/stress_correlations.png
- analysis_output/movement_analysis.png
- analysis_output/formation_analysis.png
- analysis_output/time_series_analysis.png
- analysis_output/summary_statistics.txt
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server compatibility

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr
import warnings
import os
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = 'C:\\Users\\Usuario\\Documents\\devops\\d\\BigDataBowl\\animation_data_2023090700_101.csv'
OUTPUT_DIR = 'analysis_output'
TEAM_THEME = 'seahawks'

# Analysis settings
GENERATE_POSITION_HEATMAPS = True
GENERATE_STRESS_CORRELATIONS = True
GENERATE_MOVEMENT_ANALYSIS = True
GENERATE_FORMATION_ANALYSIS = True
GENERATE_TIME_SERIES = True

# ============================================================================
# TEAM COLORS FOR VISUALIZATIONS
# ============================================================================
TEAM_COLORS = {
    'seahawks': {
        'primary': '#002244',
        'secondary': '#69BE28', 
        'offense': '#69BE28',
        'defense': '#A5ACAF'
    },
    'vikings': {
        'primary': '#4F2683',
        'secondary': '#FFC62F',
        'offense': '#FFC62F', 
        'defense': '#FFFFFF'
    }
}

colors = TEAM_COLORS.get(TEAM_THEME, TEAM_COLORS['seahawks'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"NFL HEATMAP & CORRELATION ANALYSIS")
print(f"{'='*60}\n")

# Load data
print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

print(f"Total rows: {len(df):,}")
print(f"Players: {df['nfl_id'].nunique()}")
print(f"Frames: {df['frame_id'].nunique()}")
print(f"Available columns: {list(df.columns)}")

# Clean data - remove projections and ensure we have valid positions
df_clean = df[~df['s'].isna() & ~df['x'].isna() & ~df['y'].isna()].copy()
print(f"Valid position records: {len(df_clean):,}")

# Calculate additional metrics
def calculate_speed(df):
    """Calculate speed magnitude from velocity components"""
    return np.sqrt(df['s']**2) if 's' in df.columns else 0

def calculate_distance_from_ball(df):
    """Calculate distance from estimated ball position (center of formation)"""
    ball_x = df.groupby('frame_id')['x'].mean()
    ball_y = df.groupby('frame_id')['y'].mean()
    
    distances = []
    for _, row in df.iterrows():
        frame_ball_x = ball_x.get(row['frame_id'], row['x'])
        frame_ball_y = ball_y.get(row['frame_id'], row['y'])
        dist = np.sqrt((row['x'] - frame_ball_x)**2 + (row['y'] - frame_ball_y)**2)
        distances.append(dist)
    return distances

def calculate_field_zones(df):
    """Categorize field positions into zones"""
    zones = []
    for _, row in df.iterrows():
        x, y = row['x'], row['y']
        
        # Vertical zones (field position)
        if x < 25:
            v_zone = "Deep_Own"
        elif x < 50:
            v_zone = "Own_Side" 
        elif x < 75:
            v_zone = "Opp_Side"
        else:
            v_zone = "Red_Zone"
            
        # Horizontal zones
        if y < 17.7:
            h_zone = "Left"
        elif y < 35.6:
            h_zone = "Center"
        else:
            h_zone = "Right"
            
        zones.append(f"{v_zone}_{h_zone}")
    return zones

# Calculate additional metrics
df_clean['speed_magnitude'] = df_clean.apply(lambda x: np.sqrt(x['s']**2) if pd.notna(x['s']) else 0, axis=1)
df_clean['distance_from_ball'] = calculate_distance_from_ball(df_clean)
df_clean['field_zone'] = calculate_field_zones(df_clean)

print(f"\nCalculated additional metrics:")
print(f"- Speed magnitude")
print(f"- Distance from ball")
print(f"- Field zones")

# ============================================================================
# 1. POSITION HEATMAPS
# ============================================================================
if GENERATE_POSITION_HEATMAPS:
    print(f"\nGenerating Position Heatmaps...")
    
    # Overall position heatmap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Field Position Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall position density
    offense_data = df_clean[df_clean['player_side'] == 'Offense']
    defense_data = df_clean[df_clean['player_side'] == 'Defense']
    
    # Draw field background
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_aspect('equal')
        
        # Yard lines
        for yard in range(10, 111, 10):
            ax.axvline(yard, color='white', alpha=0.3, linewidth=1)
        
        # Sidelines
        ax.axhline(0, color='white', linewidth=2)
        ax.axhline(53.3, color='white', linewidth=2)
        ax.axvline(0, color='white', linewidth=2)
        ax.axvline(120, color='white', linewidth=2)
        
        ax.set_facecolor('#2d5016')
    
    # Position density heatmap
    h1 = ax1.hist2d(offense_data['x'], offense_data['y'], bins=30, 
                      alpha=0.7, cmap='Greens', density=True)
    ax1.set_title('Offense Position Density', fontweight='bold')
    plt.colorbar(h1[3], ax=ax1)
    
    h2 = ax2.hist2d(defense_data['x'], defense_data['y'], bins=30,
                      alpha=0.7, cmap='Blues', density=True)
    ax2.set_title('Defense Position Density', fontweight='bold')
    plt.colorbar(h2[3], ax=ax2)
    
    # 2. Node stress heatmap
    if 'node_stress' in df_clean.columns:
        # Offense stress
        scatter1 = ax3.scatter(offense_data['x'], offense_data['y'], 
                              c=offense_data['node_stress'], 
                              s=50, alpha=0.7, cmap='Reds', 
                              vmin=0, vmax=1)
        ax3.set_title('Offense Node Stress', fontweight='bold')
        plt.colorbar(scatter1, ax=ax3)
        
        # Defense stress
        scatter2 = ax4.scatter(defense_data['x'], defense_data['y'],
                              c=defense_data['node_stress'],
                              s=50, alpha=0.7, cmap='Reds',
                              vmin=0, vmax=1)
        ax4.set_title('Defense Node Stress', fontweight='bold')
        plt.colorbar(scatter2, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/position_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Position heatmaps saved")

# ============================================================================
# 2. STRESS CORRELATION ANALYSIS
# ============================================================================
if GENERATE_STRESS_CORRELATIONS and 'node_stress' in df_clean.columns:
    print(f"\nGenerating Stress Correlation Analysis...")
    
    # Create correlation matrix
    correlation_metrics = ['node_stress', 'speed_magnitude', 'distance_from_ball']
    
    # Add directional components if available
    if 'dir' in df_clean.columns:
        correlation_metrics.append('dir')
    if 'o' in df_clean.columns:
        correlation_metrics.append('o')
        
    corr_data = df_clean[correlation_metrics].corr()
    
    # Create correlation heatmap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Node Stress Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Overall correlation
    sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax1, cbar_kws={"shrink": .8})
    ax1.set_title('Overall Correlations')
    
    # Offense vs Defense correlation comparison
    off_corr = df_clean[df_clean['player_side'] == 'Offense'][correlation_metrics].corr()
    def_corr = df_clean[df_clean['player_side'] == 'Defense'][correlation_metrics].corr()
    
    sns.heatmap(off_corr, annot=True, cmap='Greens', center=0,
                square=True, ax=ax2, cbar_kws={"shrink": .8})
    ax2.set_title('Offense Correlations')
    
    sns.heatmap(def_corr, annot=True, cmap='Blues', center=0,
                square=True, ax=ax3, cbar_kws={"shrink": .8})
    ax3.set_title('Defense Correlations')
    
    # Stress vs Position scatter
    ax4.scatter(df_clean[df_clean['player_side'] == 'Offense']['x'],
                df_clean[df_clean['player_side'] == 'Offense']['node_stress'],
                alpha=0.6, color=colors['offense'], label='Offense', s=30)
    ax4.scatter(df_clean[df_clean['player_side'] == 'Defense']['x'],
                df_clean[df_clean['player_side'] == 'Defense']['node_stress'],
                alpha=0.6, color=colors['defense'], label='Defense', s=30)
    ax4.set_xlabel('Field Position (X)')
    ax4.set_ylabel('Node Stress')
    ax4.set_title('Stress vs Field Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stress_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Stress correlation analysis saved")

# ============================================================================
# 3. MOVEMENT & SPEED ANALYSIS
# ============================================================================
if GENERATE_MOVEMENT_ANALYSIS:
    print(f"\n🏃 Generating Movement Analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Movement & Speed Analysis', fontsize=16, fontweight='bold')
    
    # Speed distribution by side
    offense_speeds = df_clean[df_clean['player_side'] == 'Offense']['speed_magnitude']
    defense_speeds = df_clean[df_clean['player_side'] == 'Defense']['speed_magnitude']
    
    ax1.hist(offense_speeds, bins=20, alpha=0.7, color=colors['offense'], 
             label='Offense', density=True)
    ax1.hist(defense_speeds, bins=20, alpha=0.7, color=colors['defense'],
             label='Defense', density=True)
    ax1.set_xlabel('Speed Magnitude')
    ax1.set_ylabel('Density')
    ax1.set_title('Speed Distribution by Side')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speed vs Stress correlation
    if 'node_stress' in df_clean.columns:
        ax2.scatter(df_clean['speed_magnitude'], df_clean['node_stress'],
                   alpha=0.6, s=30)
        
        # Calculate correlation
        corr_coef, p_value = pearsonr(df_clean['speed_magnitude'].dropna(), 
                                     df_clean['node_stress'].dropna())
        ax2.set_xlabel('Speed Magnitude')
        ax2.set_ylabel('Node Stress')
        ax2.set_title(f'Speed vs Stress (r={corr_coef:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df_clean['speed_magnitude'].dropna(), 
                      df_clean['node_stress'].dropna(), 1)
        p = np.poly1d(z)
        ax2.plot(df_clean['speed_magnitude'], p(df_clean['speed_magnitude']), 
                "r--", alpha=0.8)
    
    # Movement vectors (if direction available)
    if 'dir' in df_clean.columns:
        sample_data = df_clean.sample(min(200, len(df_clean)))  # Sample for clarity
        
        for _, row in sample_data.iterrows():
            if pd.notna(row['dir']) and pd.notna(row['speed_magnitude']):
                dx = row['speed_magnitude'] * np.cos(np.radians(row['dir'])) * 2
                dy = row['speed_magnitude'] * np.sin(np.radians(row['dir'])) * 2
                
                color = colors['offense'] if row['player_side'] == 'Offense' else colors['defense']
                ax3.arrow(row['x'], row['y'], dx, dy, 
                         head_width=1, head_length=1, 
                         fc=color, ec=color, alpha=0.6)
        
        ax3.set_xlim(0, 120)
        ax3.set_ylim(0, 53.3)
        ax3.set_title('Movement Vectors (Sample)')
        ax3.set_aspect('equal')
    
    # Zone analysis
    zone_stress = df_clean.groupby(['field_zone', 'player_side'])['node_stress'].mean().unstack()
    if zone_stress is not None:
        zone_stress.plot(kind='bar', ax=ax4, color=[colors['defense'], colors['offense']])
        ax4.set_title('Average Stress by Field Zone')
        ax4.set_xlabel('Field Zone')
        ax4.set_ylabel('Average Node Stress')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Side')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/movement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Movement analysis saved")

# ============================================================================
# 4. FORMATION ANALYSIS
# ============================================================================
if GENERATE_FORMATION_ANALYSIS:
    print(f"\nGenerating Formation Analysis...")
    
    # Calculate formation metrics by frame
    formation_data = []
    
    for frame in df_clean['frame_id'].unique():
        frame_data = df_clean[df_clean['frame_id'] == frame]
        
        offense = frame_data[frame_data['player_side'] == 'Offense']
        defense = frame_data[frame_data['player_side'] == 'Defense']
        
        if len(offense) > 0 and len(defense) > 0:
            # Calculate spread (standard deviation of positions)
            off_spread = np.std(offense['y'])
            def_spread = np.std(defense['y'])
            
            # Calculate depth (difference between max and min x positions)
            off_depth = offense['x'].max() - offense['x'].min()
            def_depth = defense['x'].max() - defense['x'].min()
            
            # Average stress
            avg_off_stress = offense['node_stress'].mean() if 'node_stress' in offense.columns else 0
            avg_def_stress = defense['node_stress'].mean() if 'node_stress' in defense.columns else 0
            
            formation_data.append({
                'frame_id': frame,
                'offense_spread': off_spread,
                'defense_spread': def_spread,
                'offense_depth': off_depth,
                'defense_depth': def_depth,
                'avg_offense_stress': avg_off_stress,
                'avg_defense_stress': avg_def_stress
            })
    
    formation_df = pd.DataFrame(formation_data)
    
    if len(formation_df) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Formation Analysis', fontsize=16, fontweight='bold')
        
        # Formation spread comparison
        ax1.plot(formation_df['frame_id'], formation_df['offense_spread'], 
                color=colors['offense'], label='Offense', linewidth=2)
        ax1.plot(formation_df['frame_id'], formation_df['defense_spread'],
                color=colors['defense'], label='Defense', linewidth=2)
        ax1.set_xlabel('Frame ID')
        ax1.set_ylabel('Formation Spread (yards)')
        ax1.set_title('Formation Width Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Formation depth comparison
        ax2.plot(formation_df['frame_id'], formation_df['offense_depth'],
                color=colors['offense'], label='Offense', linewidth=2)
        ax2.plot(formation_df['frame_id'], formation_df['defense_depth'],
                color=colors['defense'], label='Defense', linewidth=2)
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Formation Depth (yards)')
        ax2.set_title('Formation Depth Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Spread vs Stress correlation
        if 'node_stress' in df_clean.columns:
            ax3.scatter(formation_df['offense_spread'], formation_df['avg_offense_stress'],
                        color=colors['offense'], alpha=0.7, s=50, label='Offense')
            ax3.scatter(formation_df['defense_spread'], formation_df['avg_defense_stress'],
                        color=colors['defense'], alpha=0.7, s=50, label='Defense')
            ax3.set_xlabel('Formation Spread')
            ax3.set_ylabel('Average Stress')
            ax3.set_title('Formation Spread vs Stress')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Formation summary stats
        summary_stats = formation_df[['offense_spread', 'defense_spread', 
                                    'offense_depth', 'defense_depth']].describe()
        
        # Plot summary as table
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=summary_stats.round(2).values,
                          rowLabels=summary_stats.index,
                          colLabels=summary_stats.columns,
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax4.set_title('Formation Statistics Summary')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/formation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Formation analysis saved")

# ============================================================================
# 5. TIME SERIES ANALYSIS
# ============================================================================
if GENERATE_TIME_SERIES and 'node_stress' in df_clean.columns:
    print(f"\nGenerating Time Series Analysis...")
    
    # Calculate time series metrics
    time_series_data = []
    
    for frame in sorted(df_clean['frame_id'].unique()):
        frame_data = df_clean[df_clean['frame_id'] == frame]
        
        metrics = {
            'frame_id': frame,
            'total_players': len(frame_data),
            'offense_players': len(frame_data[frame_data['player_side'] == 'Offense']),
            'defense_players': len(frame_data[frame_data['player_side'] == 'Defense']),
            'avg_stress': frame_data['node_stress'].mean(),
            'max_stress': frame_data['node_stress'].max(),
            'min_stress': frame_data['node_stress'].min(),
            'stress_std': frame_data['node_stress'].std(),
            'avg_speed': frame_data['speed_magnitude'].mean(),
            'max_speed': frame_data['speed_magnitude'].max()
        }
        
        # Separate by side
        offense = frame_data[frame_data['player_side'] == 'Offense']
        defense = frame_data[frame_data['player_side'] == 'Defense']
        
        if len(offense) > 0:
            metrics['offense_avg_stress'] = offense['node_stress'].mean()
            metrics['offense_avg_speed'] = offense['speed_magnitude'].mean()
        
        if len(defense) > 0:
            metrics['defense_avg_stress'] = defense['node_stress'].mean()
            metrics['defense_avg_speed'] = defense['speed_magnitude'].mean()
        
        time_series_data.append(metrics)
    
    ts_df = pd.DataFrame(time_series_data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
    
    # Overall stress over time
    ax1.plot(ts_df['frame_id'], ts_df['avg_stress'], linewidth=2, color='red')
    ax1.fill_between(ts_df['frame_id'], 
                     ts_df['avg_stress'] - ts_df['stress_std'],
                     ts_df['avg_stress'] + ts_df['stress_std'],
                     alpha=0.3, color='red')
    ax1.set_xlabel('Frame ID')
    ax1.set_ylabel('Average Stress')
    ax1.set_title('Team Stress Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Offense vs Defense stress
    if 'offense_avg_stress' in ts_df.columns:
        ax2.plot(ts_df['frame_id'], ts_df['offense_avg_stress'], 
                color=colors['offense'], label='Offense', linewidth=2)
    if 'defense_avg_stress' in ts_df.columns:
        ax2.plot(ts_df['frame_id'], ts_df['defense_avg_stress'],
                color=colors['defense'], label='Defense', linewidth=2)
    ax2.set_xlabel('Frame ID')
    ax2.set_ylabel('Average Stress')
    ax2.set_title('Offense vs Defense Stress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Speed over time
    ax3.plot(ts_df['frame_id'], ts_df['avg_speed'], linewidth=2, color='blue')
    ax3.set_xlabel('Frame ID')
    ax3.set_ylabel('Average Speed')
    ax3.set_title('Team Speed Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Stress vs Speed relationship over time
    ax4.scatter(ts_df['avg_speed'], ts_df['avg_stress'], 
               c=ts_df['frame_id'], cmap='viridis', s=50)
    ax4.set_xlabel('Average Speed')
    ax4.set_ylabel('Average Stress')
    ax4.set_title('Speed vs Stress Evolution')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Frame ID')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Time series analysis saved")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================
print(f"\nGenerating Summary Statistics...")

summary_stats = {
    'Total Players': df_clean['nfl_id'].nunique(),
    'Total Frames': df_clean['frame_id'].nunique(),
    'Offense Players': len(df_clean[df_clean['player_side'] == 'Offense']['nfl_id'].unique()),
    'Defense Players': len(df_clean[df_clean['player_side'] == 'Defense']['nfl_id'].unique()),
}

if 'node_stress' in df_clean.columns:
    summary_stats.update({
        'Avg Node Stress': df_clean['node_stress'].mean(),
        'Max Node Stress': df_clean['node_stress'].max(),
        'Min Node Stress': df_clean['node_stress'].min(),
        'Offense Avg Stress': df_clean[df_clean['player_side'] == 'Offense']['node_stress'].mean(),
        'Defense Avg Stress': df_clean[df_clean['player_side'] == 'Defense']['node_stress'].mean(),
    })

summary_stats.update({
    'Avg Speed': df_clean['speed_magnitude'].mean(),
    'Max Speed': df_clean['speed_magnitude'].max(),
    'Field Coverage X': f"{df_clean['x'].min():.1f} - {df_clean['x'].max():.1f}",
    'Field Coverage Y': f"{df_clean['y'].min():.1f} - {df_clean['y'].max():.1f}",
})

# Save summary to text file
with open(f'{OUTPUT_DIR}/summary_statistics.txt', 'w') as f:
    f.write("NFL ANIMATION DATA - SUMMARY STATISTICS\n")
    f.write("="*50 + "\n\n")
    
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")
    
    f.write(f"\nField Zones Distribution:\n")
    zone_counts = df_clean['field_zone'].value_counts()
    for zone, count in zone_counts.items():
        f.write(f"  {zone}: {count}\n")

print(f"\n{'='*60}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*60}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Generated files:")
if GENERATE_POSITION_HEATMAPS:
    print(f"   position_heatmaps.png")
if GENERATE_STRESS_CORRELATIONS:
    print(f"   stress_correlations.png") 
if GENERATE_MOVEMENT_ANALYSIS:
    print(f"   movement_analysis.png")
if GENERATE_FORMATION_ANALYSIS:
    print(f"   formation_analysis.png")
if GENERATE_TIME_SERIES:
    print(f"   time_series_analysis.png")
print(f"   summary_statistics.txt")

print(f"\nKey Insights:")
for key, value in list(summary_stats.items())[:8]:
    print(f"  • {key}: {value}")
