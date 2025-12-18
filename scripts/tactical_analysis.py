#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 14: TACTICAL ANALYSIS (LIMITED COVERAGE)
==============================================

SUMMARY
-------
This script generates strategic insights from NFL tracking data, specifically 
optimized for scenarios where player tracking is incomplete (less than 22 players).
Instead of relying on perfect 11v11 modeling, it uses probabilistic density 
and zone analysis to identify tactical patterns.

It focuses on:
1. Identifying under-defended areas (Coverage Gaps).
2. Tracking formation shifts without full rosters (Formation Evolution).
3. Analyzing player movements across key field areas (Tactical Zones).

METHODOLOGY
-----------
1. Coverage Density Mapping: Uses 2D spatial binning to visualize player 
   concentration and highlight "low coverage" rectangles on the field.
2. Formation Centroid Tracking: Calculates the center of mass and standard 
   deviation (spread) of visible players to quantify formation width and depth changes.
3. Zone Logic: Maps field coordinates (x, y) to tactical zones (e.g., "Red Zone_Left", 
   "Midfield_Center")  to track territory occupation.
4. Transition Analysis: Logs sequences of zone changes to infer player movement 
   patterns (e.g., dropping back from Line of Scrimmage to Deep Zone).

INPUTS
------
1. animation_data_*.csv (Input CSV with partial tracking data)

OUTPUTS
-------
- tactical_analysis/coverage_analysis.png (Heatmaps of player density)
- tactical_analysis/formation_evolution.png (Charts of formation spread/depth)
- tactical_analysis/tactical_zones.png (Bar charts of zone occupation)
- tactical_analysis/tactical_insights.txt (Text summary of coverage % and stability)
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = 'C:\\Users\\Usuario\\Documents\\devops\\d\\BigDataBowl\\animation_data_2023090700_101.csv'
OUTPUT_DIR = 'tactical_analysis'
TEAM_THEME = 'seahawks'

colors = {'offense': '#69BE28', 'defense': '#A5ACAF', 'primary': '#002244'}
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"NFL TACTICAL ANALYSIS")
print(f"Limited Player Coverage Optimization")
print(f"{'='*60}\n")

# Load data
df = pd.read_csv(INPUT_CSV)
df_clean = df[~df['s'].isna() & ~df['x'].isna() & ~df['y'].isna()].copy()

print(f"Coverage: {df_clean['nfl_id'].nunique()}/22 players ({df_clean['nfl_id'].nunique()/22*100:.1f}%)")
print(f"Frames: {df_clean['frame_id'].nunique()}")

def draw_field_background(ax, show_zones=True):
    """Draw field with tactical zones"""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal')
    ax.set_facecolor('#2d5016')
    
    # Basic field markings
    for yard in range(10, 111, 10):
        ax.axvline(yard, color='white', alpha=0.4, linewidth=1)
    
    ax.axvline(10, color='white', linewidth=3)  # Goal line
    ax.axvline(110, color='white', linewidth=3)  # Goal line
    ax.axhline(0, color='white', linewidth=2)   # Sideline
    ax.axhline(53.3, color='white', linewidth=2)  # Sideline
    
    if show_zones:
        # Add tactical zones
        # Red zone
        ax.axvspan(0, 20, alpha=0.1, color='red', label='Red Zone')
        ax.axvspan(100, 120, alpha=0.1, color='red')
        
        # Middle field
        ax.axvspan(40, 80, alpha=0.05, color='yellow', label='Middle Field')
        
        # Hash marks (approximate)
        ax.axhline(23.36, color='white', alpha=0.3, linestyle='--')
        ax.axhline(29.94, color='white', alpha=0.3, linestyle='--')

def analyze_coverage_gaps():
    """Identify areas with limited or missing coverage"""
    
    print(f"Analyzing Coverage Gaps...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Coverage Gap Analysis', fontsize=16, fontweight='bold')
    
    # 1. Coverage density heatmap
    draw_field_background(ax1, show_zones=False)
    
    # Create grid for coverage analysis
    x_bins = np.linspace(0, 120, 25)
    y_bins = np.linspace(0, 53.3, 15)
    
    coverage_counts = np.zeros((len(y_bins)-1, len(x_bins)-1))
    
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            x_mask = (df_clean['x'] >= x_bins[i]) & (df_clean['x'] < x_bins[i+1])
            y_mask = (df_clean['y'] >= y_bins[j]) & (df_clean['y'] < y_bins[j+1])
            coverage_counts[j, i] = len(df_clean[x_mask & y_mask])
    
    im1 = ax1.imshow(coverage_counts, extent=[0, 120, 0, 53.3], 
                      cmap='Reds', alpha=0.7, aspect='auto', origin='lower')
    ax1.set_title('Player Position Density')
    plt.colorbar(im1, ax=ax1, label='Observation Count')
    
    # 2. Missing positions (low coverage areas)
    draw_field_background(ax2, show_zones=False)
    
    # Identify low coverage areas
    low_coverage_mask = coverage_counts < np.percentile(coverage_counts, 25)
    
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            if low_coverage_mask[j, i] and coverage_counts[j, i] > 0:
                rect = patches.Rectangle((x_bins[i], y_bins[j]), 
                                       x_bins[i+1] - x_bins[i], 
                                       y_bins[j+1] - y_bins[j],
                                       linewidth=1, edgecolor='red', 
                                       facecolor='red', alpha=0.3)
                ax2.add_patch(rect)
    
    ax2.set_title('Low Coverage Areas (Bottom 25%)')
    
    # 3. Player distribution by side
    offense_data = df_clean[df_clean['player_side'] == 'Offense']
    defense_data = df_clean[df_clean['player_side'] == 'Defense']
    
    ax3.hist2d(offense_data['x'], offense_data['y'], bins=15, 
               alpha=0.6, cmap='Greens')
    ax3.set_title(f'Offense Coverage ({len(offense_data["nfl_id"].unique())} players)')
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 53.3)
    
    ax4.hist2d(defense_data['x'], defense_data['y'], bins=15,
               alpha=0.6, cmap='Blues')
    ax4.set_title(f'Defense Coverage ({len(defense_data["nfl_id"].unique())} players)')
    ax4.set_xlim(0, 120)
    ax4.set_ylim(0, 53.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/coverage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return coverage_counts

def analyze_formation_changes():
    """Analyze how formations change over time with limited data"""
    
    print(f"Analyzing Formation Evolution...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Formation Analysis (Partial Coverage)', fontsize=16, fontweight='bold')
    
    # Calculate formation metrics by frame
    formation_metrics = []
    
    for frame in sorted(df_clean['frame_id'].unique()):
        frame_data = df_clean[df_clean['frame_id'] == frame]
        
        if len(frame_data) > 0:
            # Calculate center of mass
            center_x = frame_data['x'].mean()
            center_y = frame_data['y'].mean()
            
            # Calculate spread
            spread_x = frame_data['x'].std()
            spread_y = frame_data['y'].std()
            
            # Calculate by side if both sides present
            offense = frame_data[frame_data['player_side'] == 'Offense']
            defense = frame_data[frame_data['player_side'] == 'Defense']
            
            metrics = {
                'frame': frame,
                'center_x': center_x,
                'center_y': center_y,
                'spread_x': spread_x,
                'spread_y': spread_y,
                'total_players': len(frame_data),
                'offense_players': len(offense),
                'defense_players': len(defense)
            }
            
            if len(offense) > 0:
                metrics['offense_center_x'] = offense['x'].mean()
                metrics['offense_center_y'] = offense['y'].mean()
                metrics['offense_spread_y'] = offense['y'].std()
                
            if len(defense) > 0:
                metrics['defense_center_x'] = defense['x'].mean()
                metrics['defense_center_y'] = defense['y'].mean()
                metrics['defense_spread_y'] = defense['y'].std()
                
            if 'node_stress' in frame_data.columns:
                metrics['avg_stress'] = frame_data['node_stress'].mean()
                if len(offense) > 0:
                    metrics['offense_stress'] = offense['node_stress'].mean()
                if len(defense) > 0:
                    metrics['defense_stress'] = defense['node_stress'].mean()
            
            formation_metrics.append(metrics)
    
    formation_df = pd.DataFrame(formation_metrics)
    
    if len(formation_df) > 0:
        # 1. Formation center movement
        draw_field_background(ax1)
        ax1.plot(formation_df['center_x'], formation_df['center_y'], 
                 'ko-', linewidth=2, markersize=8, alpha=0.7, label='Formation Center')
        
        # Add arrows to show direction
        for i in range(len(formation_df)-1):
            dx = formation_df.iloc[i+1]['center_x'] - formation_df.iloc[i]['center_x']
            dy = formation_df.iloc[i+1]['center_y'] - formation_df.iloc[i]['center_y']
            ax1.arrow(formation_df.iloc[i]['center_x'], formation_df.iloc[i]['center_y'],
                      dx, dy, head_width=1, head_length=1, fc='black', ec='black', alpha=0.5)
        
        ax1.set_title('Formation Center Movement')
        ax1.legend()
        
        # 2. Formation width over time
        ax2.plot(formation_df['frame'], formation_df['spread_y'], 
                 'b-', linewidth=2, label='Overall Width')
        
        if 'offense_spread_y' in formation_df.columns:
            ax2.plot(formation_df['frame'], formation_df['offense_spread_y'],
                     color=colors['offense'], linewidth=2, label='Offense Width')
        if 'defense_spread_y' in formation_df.columns:
            ax2.plot(formation_df['frame'], formation_df['defense_spread_y'],
                     color=colors['defense'], linewidth=2, label='Defense Width')
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Formation Width (yards)')
        ax2.set_title('Formation Width Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Player count over time
        ax3.plot(formation_df['frame'], formation_df['total_players'], 
                 'k-', linewidth=2, label='Total Tracked')
        ax3.plot(formation_df['frame'], formation_df['offense_players'],
                 color=colors['offense'], linewidth=2, label='Offense')
        ax3.plot(formation_df['frame'], formation_df['defense_players'],
                 color=colors['defense'], linewidth=2, label='Defense')
        
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Players Tracked')
        ax3.set_title('Player Coverage Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Stress vs formation metrics (if available)
        if 'avg_stress' in formation_df.columns:
            scatter = ax4.scatter(formation_df['spread_y'], formation_df['avg_stress'],
                                c=formation_df['frame'], cmap='viridis', s=60, alpha=0.7)
            ax4.set_xlabel('Formation Width')
            ax4.set_ylabel('Average Stress')
            ax4.set_title('Formation Width vs Stress')
            plt.colorbar(scatter, ax=ax4, label='Frame')
            ax4.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/formation_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return formation_df

def analyze_tactical_zones():
    """Analyze player activity in different tactical zones"""
    
    print(f"Analyzing Tactical Zones...")
    
    # Define tactical zones
    def get_tactical_zone(x, y):
        # Vertical zones
        if x <= 20:
            v_zone = "Own_End"
        elif x <= 40:
            v_zone = "Own_Side"
        elif x <= 60:
            v_zone = "Midfield"
        elif x <= 80:
            v_zone = "Opp_Side"
        else:
            v_zone = "Red_Zone"
        
        # Horizontal zones
        if y <= 17.77:
            h_zone = "Left"
        elif y <= 35.53:
            h_zone = "Center"
        else:
            h_zone = "Right"
        
        return f"{v_zone}_{h_zone}"
    
    df_clean['tactical_zone'] = df_clean.apply(lambda row: get_tactical_zone(row['x'], row['y']), axis=1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tactical Zone Analysis', fontsize=16, fontweight='bold')
    
    # 1. Zone occupation by side
    zone_counts = df_clean.groupby(['tactical_zone', 'player_side']).size().unstack(fill_value=0)
    
    zone_counts.plot(kind='bar', ax=ax1, color=[colors['defense'], colors['offense']])
    ax1.set_title('Zone Occupation by Side')
    ax1.set_xlabel('Tactical Zone')
    ax1.set_ylabel('Observation Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Side')
    
    # 2. Average stress by zone (if available)
    if 'node_stress' in df_clean.columns:
        zone_stress = df_clean.groupby(['tactical_zone', 'player_side'])['node_stress'].mean().unstack()
        
        zone_stress.plot(kind='bar', ax=ax2, color=[colors['defense'], colors['offense']])
        ax2.set_title('Average Stress by Zone')
        ax2.set_xlabel('Tactical Zone')
        ax2.set_ylabel('Average Node Stress')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Side')
    
    # 3. Zone transition analysis
    # Track how players move between zones
    transitions = {}
    
    for player in df_clean['nfl_id'].unique():
        player_data = df_clean[df_clean['nfl_id'] == player].sort_values('frame_id')
        for i in range(len(player_data) - 1):
            current_zone = player_data.iloc[i]['tactical_zone']
            next_zone = player_data.iloc[i+1]['tactical_zone']
            
            if current_zone != next_zone:
                transition = f"{current_zone} -> {next_zone}"
                transitions[transition] = transitions.get(transition, 0) + 1
    
    if transitions:
        # Show top transitions
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        transition_names = [t[0] for t in sorted_transitions]
        transition_counts = [t[1] for t in sorted_transitions]
        
        ax3.barh(range(len(transition_names)), transition_counts)
        ax3.set_yticks(range(len(transition_names)))
        ax3.set_yticklabels(transition_names, fontsize=8)
        ax3.set_xlabel('Transition Count')
        ax3.set_title('Top Zone Transitions')
    
    # 4. Zone coverage completeness
    all_zones = [f"{v}_{h}" for v in ["Own_End", "Own_Side", "Midfield", "Opp_Side", "Red_Zone"]
                 for h in ["Left", "Center", "Right"]]
    
    covered_zones = set(df_clean['tactical_zone'].unique())
    missing_zones = set(all_zones) - covered_zones
    
    zone_coverage = []
    for zone in all_zones:
        if zone in covered_zones:
            count = len(df_clean[df_clean['tactical_zone'] == zone])
            zone_coverage.append(count)
        else:
            zone_coverage.append(0)
    
    bars = ax4.bar(range(len(all_zones)), zone_coverage)
    ax4.set_xticks(range(len(all_zones)))
    ax4.set_xticklabels(all_zones, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Observation Count')
    ax4.set_title(f'Zone Coverage ({len(covered_zones)}/{len(all_zones)} zones covered)')
    
    # Color bars based on coverage
    for i, (bar, count) in enumerate(zip(bars, zone_coverage)):
        if count == 0:
            bar.set_color('red')
            bar.set_alpha(0.3)
        elif count < 5:
            bar.set_color('orange')
            bar.set_alpha(0.6)
        else:
            bar.set_color('green')
            bar.set_alpha(0.8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tactical_zones.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_tactical_insights_summary(coverage_counts, formation_df):
    """Generate summary insights for tactical analysis"""
    
    print(f"Generating Tactical Summary...")
    
    # Calculate key insights
    insights = {
        'coverage_percentage': (df_clean['nfl_id'].nunique() / 22) * 100,
        'total_observations': len(df_clean),
        'frames_analyzed': df_clean['frame_id'].nunique(),
        'offense_players_tracked': len(df_clean[df_clean['player_side'] == 'Offense']['nfl_id'].unique()),
        'defense_players_tracked': len(df_clean[df_clean['player_side'] == 'Defense']['nfl_id'].unique()),
    }
    
    if 'node_stress' in df_clean.columns:
        insights['avg_stress'] = df_clean['node_stress'].mean()
        insights['max_stress'] = df_clean['node_stress'].max()
        insights['stress_variance'] = df_clean['node_stress'].var()
    
    # Field coverage analysis
    field_x_coverage = (df_clean['x'].max() - df_clean['x'].min()) / 120 * 100
    field_y_coverage = (df_clean['y'].max() - df_clean['y'].min()) / 53.3 * 100
    
    insights['field_x_coverage'] = field_x_coverage
    insights['field_y_coverage'] = field_y_coverage
    
    # Formation insights
    if len(formation_df) > 0:
        insights['formation_stability_x'] = formation_df['spread_x'].std()
        insights['formation_stability_y'] = formation_df['spread_y'].std()
        insights['avg_formation_width'] = formation_df['spread_y'].mean()
    
    # Save insights
    with open(f'{OUTPUT_DIR}/tactical_insights.txt', 'w') as f:
        f.write("NFL TACTICAL ANALYSIS - KEY INSIGHTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("COVERAGE ANALYSIS:\n")
        f.write(f"  • Player Coverage: {insights['coverage_percentage']:.1f}% ({df_clean['nfl_id'].nunique()}/22 players)\n")
        f.write(f"  • Offense Players Tracked: {insights['offense_players_tracked']}\n")
        f.write(f"  • Defense Players Tracked: {insights['defense_players_tracked']}\n")
        f.write(f"  • Total Observations: {insights['total_observations']:,}\n")
        f.write(f"  • Frames Analyzed: {insights['frames_analyzed']}\n\n")
        
        f.write("FIELD COVERAGE:\n")
        f.write(f"  • X-axis Coverage: {insights['field_x_coverage']:.1f}% of field length\n")
        f.write(f"  • Y-axis Coverage: {insights['field_y_coverage']:.1f}% of field width\n\n")
        
        if 'avg_stress' in insights:
            f.write("STRESS ANALYSIS:\n")
            f.write(f"  • Average Node Stress: {insights['avg_stress']:.3f}\n")
            f.write(f"  • Maximum Stress Recorded: {insights['max_stress']:.3f}\n")
            f.write(f"  • Stress Variance: {insights['stress_variance']:.3f}\n\n")
        
        if len(formation_df) > 0:
            f.write("FORMATION ANALYSIS:\n")
            f.write(f"  • Average Formation Width: {insights['avg_formation_width']:.2f} yards\n")
            f.write(f"  • Formation X Stability: {insights['formation_stability_x']:.2f}\n")
            f.write(f"  • Formation Y Stability: {insights['formation_stability_y']:.2f}\n\n")
        
        f.write("RECOMMENDATIONS FOR LIMITED COVERAGE:\n")
        f.write("  • Focus on relative positioning rather than absolute formations\n")
        f.write("  • Use stress patterns to identify key moments\n")
        f.write("  • Analyze zone transitions for tactical insights\n")
        f.write("  • Consider formation stability as a measure of discipline\n")
        f.write("  • Use partial coverage to understand player roles\n")

# Run all tactical analyses
coverage_data = analyze_coverage_gaps()
formation_data = analyze_formation_changes()
analyze_tactical_zones()
generate_tactical_insights_summary(coverage_data, formation_data)

print(f"\n{'='*60}")
print(f"TACTICAL ANALYSIS COMPLETE!")
print(f"{'='*60}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Generated files:")

tactical_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.txt'))]
for file in sorted(tactical_files):
    print(f"  ✓ {file}")

print(f"\nTactical Insights Generated:")
print(f"  • Coverage gap analysis")
print(f"  • Formation evolution tracking") 
print(f"  • Tactical zone occupation")
print(f"  • Player transition patterns")
print(f"  • Strategic recommendations for partial coverage")
