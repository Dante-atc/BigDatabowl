# coding: utf-8

"""
Advanced Defensive Performance Map (Isoquants + Pareto + Hulls) - WITH TEAM LOGOS
==================================================================================

This script generates a competition-quality defensive landscape map.
It goes beyond simple scatter plots by adding:
1. Efficiency Isoquants: Contour lines showing equal 'Total Score'.
2. Pareto Frontier: Connects the dominant teams (no one is better in both metrics).
3. Convex Hull: Visually groups the top-tier defenses.
4. Volume Sizing: Point size reflects sample size (play count).
5. TEAM LOGOS: Uses nfl_data_py to display team logos instead of dots.

INPUTS
------
1. metrics_playlevel_supervised.parquet
2. supplementary_data.csv

OUTPUTS
-------
1. defensive_performance_advanced.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import nfl_data_py as nfl
from PIL import Image
import requests
from io import BytesIO
import urllib3

# Disable SSL warnings (needed for servers with certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------
# Update these paths to match your server environment
METRICS_DIR = "/lustre/proyectos/p037/metrics"
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

METRICS_PATH = os.path.join(METRICS_DIR, "metrics_playlevel_supervised.parquet")
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")
OUTPUT_FILE = os.path.join(METRICS_DIR, "defensive_performance_advanced.png")

# Analysis Constants
TOP_N_LABELS = 10         # Number of top teams to label
WEIGHT_DCI = 0.5          # Weight for DCI in Total Score
WEIGHT_DIS = 0.5          # Weight for DIS in Total Score

# Logo Constants
LOGO_ZOOM_BASE = 0.20     # Base size for logos (increased for better visibility)
LOGO_ZOOM_SCALE = 0.10    # Additional scaling based on play count

# -------------------------------------------------------
# 2. LOAD NFL TEAM DATA (LOGOS)
# -------------------------------------------------------
print("Loading NFL team data and logos...")

# Create directory for cached logos
LOGO_CACHE_DIR = "/lustre/proyectos/p037/logos"
os.makedirs(LOGO_CACHE_DIR, exist_ok=True)

# Import teams data which includes logo URLs
teams_df = nfl.import_team_desc()

# Create a dictionary mapping team abbreviations to logo URLs
team_logo_map_wiki = dict(zip(teams_df['team_abbr'], teams_df['team_logo_wikipedia']))
team_logo_map_espn = dict(zip(teams_df['team_abbr'], teams_df['team_logo_espn']))

# Cache for loaded logo images (to avoid re-loading)
logo_image_cache = {}

# Headers to avoid 403 errors
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def normalize_logo_size(img, base_size=200):
    """
    Normalize logo to a consistent size while maintaining aspect ratio.
    """
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get current size
    width, height = img.size
    
    # Calculate scaling to fit within base_size x base_size
    scale = min(base_size / width, base_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize with high-quality resampling
    # Note: Use Image.LANCZOS for Pillow >= 9.0.0, else Image.ANTIALIAS
    if hasattr(Image, 'Resampling'):
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.LANCZOS

    img_resized = img.resize((new_width, new_height), resample_method)
    
    return img_resized

def create_fallback_logo(team_abbr, size=200):
    """
    Create a simple text-based logo for teams that couldn't be downloaded.
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a new image with transparent background
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a circle
    circle_color = (100, 100, 100, 230)  # Gray with some transparency
    draw.ellipse([10, 10, size-10, size-10], fill=circle_color, outline=(50, 50, 50, 255), width=3)
    
    # Add team abbreviation text
    try:
        # Try to use a larger font
        font = ImageFont.truetype("arial.ttf", size // 4)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), team_abbr, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    # Draw text in white
    draw.text((x, y), team_abbr, fill=(255, 255, 255, 255), font=font)
    
    return img

def get_logo_image(team_abbr, cache):
    """
    Download and return a PIL Image of the team logo.
    Uses local file caching and proper headers to avoid 403 errors.
    Returns normalized logos with consistent sizing.
    """
    # Check memory cache first
    if team_abbr in cache:
        return cache[team_abbr]
    
    # Check if we have a cached file
    cache_file = os.path.join(LOGO_CACHE_DIR, f"{team_abbr}.png")
    if os.path.exists(cache_file):
        try:
            img = Image.open(cache_file)
            img_normalized = normalize_logo_size(img)
            cache[team_abbr] = img_normalized
            return img_normalized
        except Exception as e:
            print(f"Warning: Could not load cached logo for {team_abbr}: {e}")
    
    # Try to download from Wikipedia first
    try:
        url = team_logo_map_wiki.get(team_abbr)
        if url:
            response = requests.get(url, timeout=10, verify=False, headers=HEADERS)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Normalize size before caching
                img_normalized = normalize_logo_size(img)
                # Save normalized version to cache
                img_normalized.save(cache_file, 'PNG')
                cache[team_abbr] = img_normalized
                print(f"✓ Downloaded logo for {team_abbr}")
                return img_normalized
    except Exception as e:
        print(f"Warning: Wikipedia failed for {team_abbr}: {e}")
    
    # Try ESPN as fallback
    try:
        url = team_logo_map_espn.get(team_abbr)
        if url:
            response = requests.get(url, timeout=10, verify=False, headers=HEADERS)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Normalize size before caching
                img_normalized = normalize_logo_size(img)
                # Save normalized version to cache
                img_normalized.save(cache_file, 'PNG')
                cache[team_abbr] = img_normalized
                print(f"✓ Downloaded logo for {team_abbr} (ESPN)")
                return img_normalized
    except Exception as e:
        print(f"Warning: ESPN also failed for {team_abbr}: {e}")
    
    # Create fallback logo
    print(f"⚠ Creating fallback logo for {team_abbr}")
    fallback = create_fallback_logo(team_abbr)
    # Save fallback to cache
    fallback.save(cache_file, 'PNG')
    cache[team_abbr] = fallback
    return fallback

# -------------------------------------------------------
# 3. DATA LOADING & PROCESSING
# -------------------------------------------------------
print("Loading data...")
try:
    df_metrics = pd.read_parquet(METRICS_PATH)
    df_supp = pd.read_csv(SUPP_PATH, low_memory=False)
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# Standardize Columns
df_supp.rename(columns={
    "gameId": "game_id", "playId": "play_id", "defensiveTeam": "defensive_team"
}, inplace=True)

# Merge
print("Merging datasets...")
merged_df = df_metrics.merge(
    df_supp[["game_id", "play_id", "defensive_team"]],
    on=["game_id", "play_id"],
    how="inner"
)

# Aggregation: Mean metrics AND Count (for sizing)
team_stats = merged_df.groupby("defensive_team").agg(
    dci_supervised=("dci_supervised", "mean"),
    dis_final=("dis_final", "mean"),
    play_count=("play_id", "count")
).reset_index()

# -------------------------------------------------------
# 4. SCORING & Z-SCORES
# -------------------------------------------------------
# Calculate Z-scores
team_stats["dci_z"] = (team_stats["dci_supervised"] - team_stats["dci_supervised"].mean()) / team_stats["dci_supervised"].std()
team_stats["dis_z"] = (team_stats["dis_final"] - team_stats["dis_final"].mean()) / team_stats["dis_final"].std()

# Calculate Elite Score (Weighted Sum of Z-scores)
team_stats["elite_score"] = (team_stats["dci_z"] * WEIGHT_DCI) + (team_stats["dis_z"] * WEIGHT_DIS)

# Sort
team_stats = team_stats.sort_values("elite_score", ascending=False).reset_index(drop=True)
team_stats["rank"] = team_stats.index + 1

# -------------------------------------------------------
# 5. PARETO FRONTIER CALCULATION
# -------------------------------------------------------
def get_pareto_frontier(df, x_col, y_col):
    # Sort by X descending
    sorted_df = df.sort_values(x_col, ascending=False)
    pareto_front = []
    
    # Track the maximum Y seen so far
    max_y = -np.inf
    
    for row in sorted_df.itertuples():
        # If this point has a higher Y than any X-greater point before it, it's on the frontier
        if getattr(row, y_col) >= max_y:
            pareto_front.append((getattr(row, x_col), getattr(row, y_col)))
            max_y = getattr(row, y_col)
            
    # Sort the frontier points by X again to plot a clean line
    return sorted(pareto_front, key=lambda x: x[0])

pareto_points = get_pareto_frontier(team_stats, "dci_supervised", "dis_final")
pareto_x, pareto_y = zip(*pareto_points) if pareto_points else ([], [])

# -------------------------------------------------------
# 6. PLOTTING
# -------------------------------------------------------
print("Creating visualization...")
plt.figure(figsize=(14, 11))
ax = plt.gca()

# --- A. Isoquants (Contour Lines) ---
# Create a grid covering the data range
x_min, x_max = team_stats["dci_supervised"].min(), team_stats["dci_supervised"].max()
y_min, y_max = team_stats["dis_final"].min(), team_stats["dis_final"].max()
pad_x, pad_y = (x_max - x_min)*0.1, (y_max - y_min)*0.1

xi = np.linspace(x_min - pad_x, x_max + pad_x, 100)
yi = np.linspace(y_min - pad_y, y_max + pad_y, 100)
X, Y = np.meshgrid(xi, yi)

# Reconstruct score logic for grid
mu_dci, sig_dci = team_stats["dci_supervised"].mean(), team_stats["dci_supervised"].std()
mu_dis, sig_dis = team_stats["dis_final"].mean(), team_stats["dis_final"].std()

Z_dci = (X - mu_dci) / sig_dci
Z_dis = (Y - mu_dis) / sig_dis
Z_score = (Z_dci * WEIGHT_DCI) + (Z_dis * WEIGHT_DIS)

# Plot Contours
levels = np.linspace(Z_score.min(), Z_score.max(), 8)
cntr = plt.contour(X, Y, Z_score, levels=levels, colors='gray', alpha=0.2, linestyles='dashed')
plt.clabel(cntr, inline=True, fontsize=8, fmt='Score: %.1f')

# --- B. Convex Hull (Elite Tier) ---
# Select top 8 teams for the "Elite Cluster"
top_tier = team_stats.head(8)[["dci_supervised", "dis_final"]].values
if len(top_tier) >= 3:
    hull = ConvexHull(top_tier)
    # Get the hull vertices in order
    hull_points = top_tier[hull.vertices]
    # Close the loop
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)
    
    plt.fill(hull_points[:,0], hull_points[:,1], color='gold', alpha=0.1, label='Elite Tier')
    plt.plot(hull_points[:,0], hull_points[:,1], color='gold', alpha=0.4, linestyle='--')

# --- C. TEAM LOGOS (Replacing Scatter Plot) with Collision Avoidance ---
print("Adding team logos with collision avoidance...")

# Normalize play count for sizing
min_count = team_stats["play_count"].min()
max_count = team_stats["play_count"].max()

# First pass: calculate positions and sizes
logo_data = []
for idx, row in team_stats.iterrows():
    team_abbr = row["defensive_team"]
    x = row["dci_supervised"]
    y = row["dis_final"]
    
    # Get logo image (always returns an image, either real or fallback)
    logo_img = get_logo_image(team_abbr, logo_image_cache)
    
    # Scale logo size based on play count
    size_scale = (row["play_count"] - min_count) / (max_count - min_count) if max_count > min_count else 0.5
    zoom = LOGO_ZOOM_BASE + (size_scale * LOGO_ZOOM_SCALE)
    
    logo_data.append({
        'team': team_abbr,
        'x_orig': x,
        'y_orig': y,
        'x': x,  # Current position (will be adjusted)
        'y': y,  # Current position (will be adjusted)
        'logo': logo_img,
        'zoom': zoom,
        'elite_score': row["elite_score"]
    })

# Helper function to estimate logo size in data coordinates
def estimate_logo_box(logo_dict, ax, fig):
    """
    Estimate the bounding box of a logo in data coordinates.
    FIX: Uses axes width (not figure width) for accurate collision detection.
    """
    zoom = logo_dict['zoom']
    
    # Get axis bounds
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    # Get figure dimensions and axes position (bbox)
    fig_width_inches = fig.get_figwidth()
    fig_height_inches = fig.get_figheight()
    
    # Get axes position in relative figure coordinates (0-1)
    bbox = ax.get_position()
    ax_width_inches = fig_width_inches * bbox.width
    ax_height_inches = fig_height_inches * bbox.height
    
    dpi = fig.dpi
    
    # Logo pixel size (approximate based on 200px base image)
    logo_pixel_size = 200 * zoom
    
    # Convert pixels to data units using AXES dimensions
    # Padding factor of 1.2 adds buffer zone
    data_width = (logo_pixel_size / (ax_width_inches * dpi)) * x_range * 1.2
    data_height = (logo_pixel_size / (ax_height_inches * dpi)) * y_range * 1.2
    
    return data_width, data_height

# Collision detection and resolution
def check_collision(logo1, logo2, box1, box2):
    """Check if two logos collide."""
    w1, h1 = box1
    w2, h2 = box2
    
    # Check if bounding boxes overlap
    dx = abs(logo1['x'] - logo2['x'])
    dy = abs(logo1['y'] - logo2['y'])
    
    return dx < (w1 + w2) / 2 and dy < (h1 + h2) / 2

def resolve_collisions(logo_data, ax, fig, max_iterations=100):
    """
    Iteratively adjust logo positions to avoid overlaps.
    """
    for iteration in range(max_iterations):
        moved = False
        
        for i, logo1 in enumerate(logo_data):
            box1 = estimate_logo_box(logo1, ax, fig)
            
            for j, logo2 in enumerate(logo_data):
                if i >= j:
                    continue
                
                box2 = estimate_logo_box(logo2, ax, fig)
                
                if check_collision(logo1, logo2, box1, box2):
                    # Calculate repulsion vector
                    dx = logo1['x'] - logo2['x']
                    dy = logo1['y'] - logo2['y']
                    
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 1e-6:  # Avoid division by zero
                        dx, dy = np.random.randn(2) * 0.001
                        dist = np.sqrt(dx**2 + dy**2)
                    
                    # Normalize and apply repulsion
                    dx /= dist
                    dy /= dist
                    
                    # Push apart with stronger force
                    push_strength = 0.0015
                    weight1 = 1.0 / (logo1['zoom'] + 0.1)
                    weight2 = 1.0 / (logo2['zoom'] + 0.1)
                    total_weight = weight1 + weight2
                    
                    logo1['x'] += dx * push_strength * (weight1 / total_weight)
                    logo1['y'] += dy * push_strength * (weight1 / total_weight)
                    logo2['x'] -= dx * push_strength * (weight2 / total_weight)
                    logo2['y'] -= dy * push_strength * (weight2 / total_weight)
                    
                    moved = True
            
            # Apply spring force to return to original position
            spring_strength = 0.15 
            logo1['x'] += (logo1['x_orig'] - logo1['x']) * spring_strength
            logo1['y'] += (logo1['y_orig'] - logo1['y']) * spring_strength
        
        if not moved:
            break
    
    print(f"  Collision resolution completed in {iteration + 1} iterations")

# Get figure for size calculations
fig = plt.gcf()

# Resolve collisions
resolve_collisions(logo_data, ax, fig)

# Second pass: place logos at adjusted positions
for logo_dict in logo_data:
    x = logo_dict['x']
    y = logo_dict['y']
    logo_img = logo_dict['logo']
    zoom = logo_dict['zoom']
    
    # Create OffsetImage
    imagebox = OffsetImage(logo_img, zoom=zoom)
    
    # Create AnnotationBbox with white border for visibility
    ab = AnnotationBbox(
        imagebox, 
        (x, y),
        frameon=True,
        pad=0.1,
        bboxprops=dict(
            edgecolor='white',
            facecolor='white',
            linewidth=2,
            boxstyle='round,pad=0.05',
            alpha=0.9
        )
    )
    ax.add_artist(ab)
    
    # Draw a subtle line connecting adjusted position to original position (if moved significantly)
    dx = abs(x - logo_dict['x_orig'])
    dy = abs(y - logo_dict['y_orig'])
    if dx > 0.0005 or dy > 0.0005:  # Only draw if moved noticeably
        ax.plot([logo_dict['x_orig'], x], [logo_dict['y_orig'], y], 
                'k--', alpha=0.15, linewidth=0.8, zorder=1)

# Add Colorbar (using a dummy scatter for the colorbar)
norm = plt.Normalize(vmin=team_stats["elite_score"].min(), vmax=team_stats["elite_score"].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Composite Elite Score (z-DCI + z-DIS)", rotation=270, labelpad=15, fontweight='bold')

# --- D. Pareto Frontier Line ---
plt.plot(pareto_x, pareto_y, color='crimson', linestyle='-', linewidth=2.5, alpha=0.6, zorder=2, label='Pareto Frontier')

# --- E. Labels & Reference Lines ---
# League Averages
plt.axvline(mu_dci, color="black", linestyle=":", alpha=0.4)
plt.axhline(mu_dis, color="black", linestyle=":", alpha=0.4)
plt.text(x_max, mu_dis, "League Avg DIS", ha="right", va="bottom", fontsize=9, alpha=0.5)
plt.text(mu_dci, y_max, "League Avg DCI", ha="left", va="top", fontsize=9, alpha=0.5, rotation=90)

# Label Top N Teams - with better positioning to avoid logo overlap
top_teams_list = team_stats.head(TOP_N_LABELS)

# Larger offsets to clear the logos, with more variation
offset_patterns = [
    (50, 50),    # Upper right
    (50, -50),   # Lower right  
    (-50, 50),   # Upper left
    (-50, -50),  # Lower left
    (0, 60),     # Directly above
    (0, -60),    # Directly below
    (60, 0),     # Directly right
    (-60, 0),    # Directly left
    (40, 40),    # Upper right (closer)
    (40, -40)    # Lower right (closer)
]

for i, row in top_teams_list.iterrows():
    dx, dy = offset_patterns[i % len(offset_patterns)]
    
    # Just show the rank number (team abbr is already in the logo)
    label_txt = f"#{row['rank']}"
    
    plt.annotate(
        label_txt,
        (row["dci_supervised"], row["dis_final"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        ha='center',
        color='black',
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", ec="orange", alpha=0.95, lw=2),
        arrowprops=dict(
            arrowstyle="->", 
            color="orange", 
            lw=2,
            alpha=0.7,
            connectionstyle="arc3,rad=0.2"
        )
    )

# -------------------------------------------------------
# 7. FINAL LAYOUT
# -------------------------------------------------------
plt.title("Defensive Landscape: The Elite Frontier", fontsize=20, fontweight="bold", pad=15)
plt.xlabel("Defensive Coverage Index (DCI) → Higher is Tighter", fontsize=13, fontweight='bold')
plt.ylabel("Defensive Integrity Score (DIS) → Higher is Better", fontsize=13, fontweight='bold')

# Add definition box
info_text = (
    "○ Logo Size: Play Volume (Sample Size)\n"
    "-- Dashed Lines: Efficiency Isoquants\n"
    "▬ Red Line: Pareto Frontier (Unbeatable Tradeoffs)\n"
    "Yellow Zone: Elite Defense Cluster"
)
plt.text(x_min, y_max, info_text, fontsize=9, va='top', ha='left',
         bbox=dict(boxstyle="round", fc="whitesmoke", ec="none", alpha=0.8))

plt.grid(True, linestyle='--', alpha=0.15)
plt.tight_layout()

# Save
print(f"Saving plot...")
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"[SUCCESS] Plot saved to: {OUTPUT_FILE}")

# Note: Comment out plt.show() if running on a headless server
# plt.show()
