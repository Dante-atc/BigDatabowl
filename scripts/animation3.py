"""
Phase 15: ENHANCED ANIMATION WITH LIVE METRICS
==============================================

SUMMARY
-------
This script generates a high-definition play animation (MP4/GIF) that 
integrates real-time defensive metrics and team-specific branding.
It visualizes the "invisible" aspects of the game: node stress, coverage 
tightness (DCI), and structural integrity (DIS).

FEATURES
--------
1. Live Metric Gauge: A heads-up display (HUD) in the corner updating DCI and 
   DIS scores frame-by-frame.
2. Stress Halos: Visual indicators around players showing isolation risk 
   (Green = Safe, Yellow = Risk, Red = Breaking Point).
3. Team Branding: Uses official hex colors for field endzones, player dots, 
   and UI elements (Ravens vs. 49ers Christmas theme enabled).
4. Player Name Resolution: Maps NFL IDs to real names (Lamar Jackson, Nick Bosa, etc.)
   for broadcast-quality labeling.

INPUTS
------
1. ravens_play_1531_data.csv (Processed tracking data with stress metrics)

OUTPUTS
-------
- team_animations/ravens_game2023122502_play1531_metrics.mp4
- team_animations/ravens_game2023122502_play1531_metrics.gif
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# TEAM COLOR SCHEMES
# ============================================================================
TEAM_COLORS = {
    'vikings': {
        'name': 'Minnesota Vikings',
        'primary': '#4F2683',
        'secondary': '#FFC62F',
        'background': '#4F2683',
        'field': '#2d5016',
        'offense_color': '#FFC62F',
        'defense_color': '#FFFFFF',
        'field_stripe': '#3d3166'
    },
    'seahawks': {
        'name': 'Seattle Seahawks',
        'primary': '#002244',
        'secondary': '#69BE28',
        'background': '#002244',
        'field': '#2d5016',
        'offense_color': '#69BE28',
        'defense_color': '#A5ACAF',
        'field_stripe': '#001933'
    },
    'ravens': {
        'name': 'Baltimore Ravens',
        'primary': '#241773',
        'secondary': '#9E7C0C',
        'background': '#241773',
        'field': '#2d5016',
        'offense_color': '#9E7C0C',
        'defense_color': '#FFFFFF',
        'field_stripe': '#1a1155'
    },
    'default': {
        'name': 'NFL Default',
        'primary': '#013369',
        'secondary': '#D50A0A',
        'background': '#0a0e1a',
        'field': '#2d5016',
        'offense_color': '#1E90FF',
        'defense_color': '#DC143C',
        'field_stripe': '#1d3010'
    }
}

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = 'ravens_play_1531_data.csv'  # Ravens vs 49ers Christmas game with stress
OUTPUT_DIR = 'team_animations'
TEAM_THEME = 'ravens'

GAME_ID = None   # Set to specific game or None to see options (not needed for new format)
PLAY_ID = None   # Set to specific play or None to see options (not needed for new format)

# Animation settings
FPS = 10
VIDEO_FORMAT = 'gif'  # 'mp4', 'gif', or 'both'
SHOW_POSITIONS = False  # Set to False to only show player names (no position labels)
SHOW_TEAM_LOGO_AREA = True
SHOW_NODE_STRESS = True
SHOW_DCI_DIS_GAUGE = True  # Show the defensive metrics gauge

# Metric calculation mode
# 'precomputed' - use DCI/DIS columns from CSV
# 'geometric' - calculate proxy metrics from spatial data
METRIC_MODE = 'geometric'  # Change to 'precomputed' if you have DCI/DIS in CSV

# Column names for precomputed metrics (if available)
DCI_COLUMN = 'dci_score'
DIS_COLUMN = 'dis_score'

# ============================================================================
# PLAYER NAME LOOKUPS
# ============================================================================

# Vikings @ 49ers October 23, 2023 Player Lookup
VIKINGS_49ERS_PLAYER_NAMES = {
    # Based on actual player IDs from vikings_play_79_data.csv
    # VIKINGS OFFENSE:
    38632: 'Jordan Addison',       # WR - Vikings rookie receiver  
    47791: 'Kirk Cousins',         # QB - Vikings starting QB
    47852: 'Justin Jefferson',     # WR - Vikings star receiver
    47885: 'T.J. Hockenson',       # TE - Vikings star tight end
    52584: 'Alexander Mattison',   # RB - Vikings running back
    55887: 'K.J. Osborn',          # WR - Vikings receiver
    
    # 49ERS DEFENSE:
    38868: 'Nick Bosa',            # DE - 49ers star pass rusher
    46139: 'Fred Warner',          # MLB - 49ers star linebacker
    46157: 'Dre Greenlaw',         # LB - 49ers linebacker
    46757: 'Javon Hargrave',       # DT - 49ers defensive tackle
    47931: 'Charvarius Ward',      # CB - 49ers cornerback
    53601: 'Talanoa Hufanga',      # SS - 49ers safety
    53609: 'Deommodore Lenoir',    # CB - 49ers cornerback
}

# Seahawks @ Giants October 2, 2023 Player Lookup
SEAHAWKS_GIANTS_PLAYER_NAMES = {
    38577: 'Bobby Wagner',        # Defense MLB - Seahawks veteran linebacker
    39987: 'Drew Lock',           # Offense QB - Seahawks backup QB
    42412: 'Jake Bobo',           # Offense WR - Seahawks receiver
    42543: 'Quandre Diggs',       # Defense FS - Seahawks safety #6
    42547: 'Will Dissly',         # Offense TE - Seahawks tight end
    43329: 'Tyler Lockett',       # Offense WR - Seahawks veteran WR #16
    43333: 'Uchenna Nwosu',       # Defense OLB - Seahawks pass rusher
    44818: 'Boye Mafe',           # Defense OLB - Seahawks edge rusher
    44830: 'Riq Woolen',          # Defense CB - Seahawks cornerback
    45186: 'Kenneth Walker III',  # Offense RB - Seahawks starting RB
    46117: 'Jarran Reed',         # Defense OLB - Seahawks defensive line
    46189: 'Darren Waller',       # Offense TE - Giants tight end
    47789: 'Geno Smith',          # Offense QB - Seahawks starting QB
    47793: 'Bobby Okereke',       # Defense MLB - Giants linebacker
    47803: 'Noah Fant',           # Offense TE - Seahawks primary TE
    47825: 'Daniel Jones',        # Offense QB - Giants starting QB
    47842: 'Darius Slayton',      # Offense WR - Giants receiver
    47847: 'DK Metcalf',          # Offense WR - Seahawks star WR #14
    47872: 'Jordyn Brooks',       # Defense ILB - Seahawks linebacker
    47891: 'Jamal Adams',         # Defense SS - Seahawks safety
    47941: 'Devon Witherspoon',   # Defense CB - Seahawks rookie CB
    47954: 'Jaxon Smith-Njigba',  # Offense WR - Seahawks rookie WR
    48266: 'Sterling Shepard',    # Offense WR - Giants receiver
    52416: 'Leonard Williams',    # Defense MLB - Seahawks DT
    52435: 'Dre\'Mont Jones',     # Defense ILB - Seahawks DE
    52444: 'Xavier McKinney',     # Defense SS - Giants safety
    52541: 'Colby Parkinson',     # Offense TE - Seahawks TE
    52552: 'Saquon Barkley',      # Offense RB - Giants star RB
    52615: 'Dareke Young',        # Offense WR - Seahawks WR
    53604: 'Julian Love',         # Defense FS - Seahawks safety
    53625: 'Zach Charbonnet',     # Offense RB - Seahawks rookie RB
    54014: 'Tre Brown',           # Defense ILB - Seahawks CB
    54470: 'Johnathan Abram',     # Defense OLB - Seahawks safety
    54506: 'DeeJay Dallas',       # Offense RB - Seahawks RB
    54508: 'Wan\'Dale Robinson',  # Offense WR - Giants receiver
    54546: 'Deonte Banks',        # Defense CB - Giants rookie CB
    54577: 'Pharaoh Brown',       # Offense TE - Giants TE
    54579: 'Isaiah Simmons',      # Defense SS - Giants LB/S
    54611: 'Micah McFadden',      # Defense ILB - Giants linebacker
    54618: 'Adoree\' Jackson',    # Defense CB - Giants cornerback
    55869: 'Nick McCloud',        # Defense CB - Giants CB
    55884: 'Jalin Hyatt',         # Offense WR - Giants rookie WR
    55888: 'Cor\'Dale Flott',     # Defense CB - Giants cornerback
    55902: 'Dane Belton',         # Defense OLB - Giants safety
    55917: 'Matt Breida',         # Offense RB - Giants RB
    55938: 'Parris Campbell',     # Offense WR - Giants receiver
    56063: 'Bobby McCain',        # Defense SS - Giants safety
    56471: 'Isaiah Hodgins',      # Offense WR - Giants receiver
}

# Ravens @ 49ers December 25, 2023 Player Lookup  
RAVENS_49ERS_PLAYER_NAMES = {
    # RAVENS OFFENSE
    54727: 'Lamar Jackson',       # QB - Ravens MVP quarterback (#8)
    44959: 'Isaiah Likely',       # TE - Ravens tight end (#80) 
    44820: 'Gus Edwards',         # RB - Ravens running back (#35)
    47819: 'Zay Flowers',         # WR - Ravens rookie receiver (#4)
    42419: 'Nelson Agholor',      # WR - Ravens receiver (#15) - scored TD
    52433: 'Rashod Bateman',      # WR - Ravens receiver (#7)
    
    # 49ERS DEFENSE
    46077: 'Fred Warner',         # MLB - 49ers star linebacker (#54)
    52436: 'Dre Greenlaw',        # ILB - 49ers linebacker (#57)
    44828: 'Charvarius Ward',     # CB - 49ers cornerback (#7)
    53533: 'Deommodore Lenoir',   # CB - 49ers cornerback (#2)
    44854: 'Ji\'Ayir Brown',      # FS - 49ers safety (#27)
    52627: 'Tashaun Gipson Sr.',  # FS - 49ers safety (#31)
    41269: 'Nick Bosa',           # OLB/DE - 49ers star pass rusher (#97)
}

# ============================================================================
# GAME CONFIGURATION - Choose which game's player names to use
# ============================================================================
GAME_MATCHUP = 'ravens_49ers'  # Options: 'vikings_49ers', 'seahawks_giants', 'ravens_49ers'

# Select the appropriate player database
if GAME_MATCHUP == 'vikings_49ers':
    PLAYER_NAMES = VIKINGS_49ERS_PLAYER_NAMES
elif GAME_MATCHUP == 'seahawks_giants':
    PLAYER_NAMES = SEAHAWKS_GIANTS_PLAYER_NAMES
elif GAME_MATCHUP == 'ravens_49ers':
    PLAYER_NAMES = RAVENS_49ERS_PLAYER_NAMES
else:
    PLAYER_NAMES = {}
    print(f"Warning: Unknown GAME_MATCHUP '{GAME_MATCHUP}'. Using empty player database.")

# Label display options
SHOW_PLAYER_NAMES = True      # Show actual names instead of IDs
SHOW_PLAYER_NUMBERS = False   # Show jersey numbers (if you have them)  
SHOW_POSITIONS = False        # Show position abbreviations (set to False to only show names)

# ============================================================================

colors = TEAM_COLORS.get(TEAM_THEME, TEAM_COLORS['default'])
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"NFL ANIMATION - {colors['name']} Theme")
print(f"{'='*60}\n")

# Load data
print(f"Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

print(f"Total rows: {len(df):,}")
print(f"Available columns: {list(df.columns)}")

# For new animation data format without game_id/play_id
if 'game_id' not in df.columns or 'play_id' not in df.columns:
    print(f"New animation data format detected (no game_id/play_id columns)")
    print(f"Players: {df['nfl_id'].nunique()}")
    print(f"Frames: {df['frame_id'].nunique()}")
    
    # Use all data as single play
    play_data = df.copy()
    GAME_ID = "animation_data"  # Dummy value for output filename
    PLAY_ID = "101"  # Dummy value for output filename
    
else:
    # Original format with game_id/play_id
    print(f"Total games: {df['game_id'].nunique()}")
    print(f"Total plays: {len(df.groupby(['game_id', 'play_id']))}")
    
    # Analyze available plays
    play_summary = df.groupby(['game_id', 'play_id']).agg({
        'frame_id': 'nunique',
        'nfl_id': 'nunique'
    }).reset_index()
    play_summary.columns = ['game_id', 'play_id', 'frames', 'players']
    play_summary = play_summary.sort_values(['players', 'frames'], ascending=[False, False])

    # If GAME_ID or PLAY_ID not set, show options
    if GAME_ID is None or PLAY_ID is None:
        print(f"\n{'='*60}")
        print("AVAILABLE PLAYS IN YOUR CSV")
        print(f"{'='*60}\n")
        print("Top 20 plays by player count:\n")
        print(play_summary.head(20).to_string(index=False))
        
        print(f"\n{'='*60}")
        print("TO GENERATE AN ANIMATION:")
        print(f"{'='*60}")
        print("1. Pick a game_id and play_id from the list above")
        print("2. Edit this script and set:")
        print("   GAME_ID = 2023090700  # (your chosen game)")
        print("   PLAY_ID = 1869         # (your chosen play)")
        print("3. Run the script again")
        exit(0)

    # Filter to selected play
    play_data = df[(df['game_id'] == GAME_ID) & (df['play_id'] == PLAY_ID)].copy()

    if len(play_data) == 0:
        print(f"\n{'='*60}")
        print(f"ERROR: No data found for Game {GAME_ID}, Play {PLAY_ID}")
        print(f"{'='*60}\n")
        exit(1)

# Check if we have precomputed metrics
has_dci = DCI_COLUMN in df.columns
has_dis = DIS_COLUMN in df.columns

if METRIC_MODE == 'precomputed' and not (has_dci and has_dis):
    print(f"\n⚠️  WARNING: METRIC_MODE set to 'precomputed' but columns not found!")
    print(f"   Looking for: {DCI_COLUMN}, {DIS_COLUMN}")
    print(f"   Available columns: {list(df.columns)}")
    print(f"   Switching to 'geometric' mode...")
    METRIC_MODE = 'geometric'

print(f"\nGame ID: {GAME_ID}")
print(f"Play ID: {PLAY_ID}")
print(f"Players: {play_data['nfl_id'].nunique()}")
print(f"Frames: {play_data['frame_id'].nunique()}")
print(f"Theme: {colors['name']}")
print(f"Metric Mode: {METRIC_MODE.upper()}")

# Separate actual positions from projections
play_data['is_projection'] = play_data['s'].isna()

def calculate_geometric_dci(defense_positions):
    """
    Calculate a geometric proxy for DCI (Coverage Tightness)
    Based on average spacing between defenders and nearest offensive player
    Returns value between 0 (loose) and 1 (tight)
    """
    if len(defense_positions) < 2:
        return 0.5
    
    # Calculate average nearest-neighbor distance among defenders
    distances = []
    for i, pos in enumerate(defense_positions):
        other_pos = np.delete(defense_positions, i, axis=0)
        if len(other_pos) > 0:
            dists = np.linalg.norm(other_pos - pos, axis=1)
            distances.append(np.min(dists))
    
    if not distances:
        return 0.5
    
    avg_spacing = np.mean(distances)
    
    # Convert to 0-1 scale (inverse: smaller spacing = higher DCI)
    # Typical spacing: 3-15 yards
    dci = np.clip(1.0 - (avg_spacing - 3.0) / 12.0, 0.0, 1.0)
    return dci

def calculate_geometric_dis(defense_positions, prev_defense_positions=None):
    """
    Calculate a geometric proxy for DIS (Structural Integrity)
    Based on variance in defender spacing (low variance = high integrity)
    Returns value between 0 (chaotic) and 1 (disciplined)
    """
    if len(defense_positions) < 3:
        return 0.5
    
    # Calculate all pairwise distances
    distances = []
    for i in range(len(defense_positions)):
        for j in range(i + 1, len(defense_positions)):
            dist = np.linalg.norm(defense_positions[i] - defense_positions[j])
            distances.append(dist)
    
    if not distances:
        return 0.5
    
    # Low coefficient of variation = consistent spacing = high integrity
    std_spacing = np.std(distances)
    mean_spacing = np.mean(distances)
    
    if mean_spacing > 0:
        cv = std_spacing / mean_spacing
        # Convert CV to 0-1 scale (lower CV = higher DIS)
        dis = np.clip(1.0 - cv, 0.0, 1.0)
    else:
        dis = 0.5
    
    return dis

def draw_field(ax, colors):
    """Draw NFL field with team-themed markings"""
    # Yard lines
    for yard in range(10, 111, 5):
        linewidth = 2 if yard % 10 == 0 else 1
        color = 'white' if yard % 10 == 0 else colors['secondary']
        alpha = 0.7 if yard % 10 == 0 else 0.4
        ax.plot([yard, yard], [0, 53.3], color=color, linewidth=linewidth, alpha=alpha)
    
    # Goal lines
    ax.plot([10, 10], [0, 53.3], color=colors['secondary'], linewidth=4, alpha=0.9)
    ax.plot([110, 110], [0, 53.3], color=colors['secondary'], linewidth=4, alpha=0.9)
    
    # Sidelines
    ax.plot([0, 120], [0, 0], color='white', linewidth=2, alpha=0.8)
    ax.plot([0, 120], [53.3, 53.3], color='white', linewidth=2, alpha=0.8)
    ax.plot([0, 0], [0, 53.3], color='white', linewidth=2, alpha=0.8)
    ax.plot([120, 120], [0, 53.3], color='white', linewidth=2, alpha=0.8)
    
    # Hash marks
    for yard in range(10, 111):
        ax.plot([yard, yard], [23.36, 23.36], color='white', marker='.', markersize=2, alpha=0.6)
        ax.plot([yard, yard], [29.94, 29.94], color='white', marker='.', markersize=2, alpha=0.6)
    
    # Striped pattern
    for yard in range(0, 120, 10):
        rect = patches.Rectangle((yard, 0), 5, 53.3, linewidth=0, 
                                 edgecolor='none', facecolor=colors['field_stripe'], alpha=0.15)
        ax.add_patch(rect)
    
    # Team branding areas
    if SHOW_TEAM_LOGO_AREA:
        top_left = patches.Rectangle((0, 48), 8, 5.3, linewidth=0,
                                    facecolor=colors['primary'], alpha=0.3)
        top_right = patches.Rectangle((112, 48), 8, 5.3, linewidth=0,
                                     facecolor=colors['secondary'], alpha=0.3)
        ax.add_patch(top_left)
        ax.add_patch(top_right)

def draw_dci_dis_gauge(ax, dci_value, dis_value, colors):
    """
    Draw DCI/DIS gauge in upper right corner with improved spacing
    Returns list of artists for animation updates
    """
    artists = []
    
    # Position in data coordinates (upper right of field) - larger and better positioned
    gauge_x = 102
    gauge_y = 43
    gauge_width = 16
    gauge_height = 9
    
    # Background box with better proportions
    bg = Rectangle((gauge_x, gauge_y), gauge_width, gauge_height,
                   facecolor=colors['primary'], edgecolor='white',
                   linewidth=2, alpha=0.9, zorder=100)
    ax.add_patch(bg)
    artists.append(bg)
    
    # Title with better spacing
    title = ax.text(gauge_x + gauge_width/2, gauge_y + gauge_height - 1,
                    'DEFENSIVE METRICS', ha='center', va='top',
                    fontsize=10, color='white', fontweight='bold', zorder=101)
    artists.append(title)
    
    # DCI Bar (Coverage Tightness) - improved spacing
    bar_y_dci = gauge_y + gauge_height - 3.5
    bar_height = 1.2
    bar_width = gauge_width - 3
    
    # DCI Background
    dci_bg = Rectangle((gauge_x + 1.5, bar_y_dci), bar_width, bar_height,
                       facecolor='#333333', edgecolor='white', 
                       linewidth=1, alpha=0.5, zorder=101)
    ax.add_patch(dci_bg)
    artists.append(dci_bg)
    
    # DCI Fill (color based on value)
    dci_color = plt.cm.RdYlGn(dci_value)  # Red=low, Yellow=mid, Green=high
    dci_fill = Rectangle((gauge_x + 1.5, bar_y_dci), bar_width * dci_value, bar_height,
                         facecolor=dci_color, edgecolor='none', alpha=0.8, zorder=102)
    ax.add_patch(dci_fill)
    artists.append(dci_fill)
    
    # DCI Label - better positioning
    dci_label = ax.text(gauge_x + 1.5, bar_y_dci - 0.4, 'DCI (Coverage)',
                        ha='left', va='top', fontsize=8, color='white',
                        fontweight='bold', zorder=103)
    artists.append(dci_label)
    
    # DCI Value - better positioning
    dci_text = ax.text(gauge_x + gauge_width - 1.5, bar_y_dci + bar_height/2,
                       f'{dci_value:.3f}', ha='right', va='center',
                       fontsize=9, color='white', fontweight='bold', zorder=103)
    artists.append(dci_text)
    
    # DIS Bar (Structural Integrity) - better spacing
    bar_y_dis = bar_y_dci - 2.5
    
    # DIS Background
    dis_bg = Rectangle((gauge_x + 1.5, bar_y_dis), bar_width, bar_height,
                       facecolor='#333333', edgecolor='white',
                       linewidth=1, alpha=0.5, zorder=101)
    ax.add_patch(dis_bg)
    artists.append(dis_bg)
    
    # DIS Fill
    dis_color = plt.cm.RdYlGn(dis_value)
    dis_fill = Rectangle((gauge_x + 1.5, bar_y_dis), bar_width * dis_value, bar_height,
                         facecolor=dis_color, edgecolor='none', alpha=0.8, zorder=102)
    ax.add_patch(dis_fill)
    artists.append(dis_fill)
    
    # DIS Label - better positioning
    dis_label = ax.text(gauge_x + 1.5, bar_y_dis - 0.4, 'DIS (Integrity)',
                        ha='left', va='top', fontsize=8, color='white',
                        fontweight='bold', zorder=103)
    artists.append(dis_label)
    
    # DIS Value - better positioning
    dis_text = ax.text(gauge_x + gauge_width - 1.5, bar_y_dis + bar_height/2,
                       f'{dis_value:.3f}', ha='right', va='center',
                       fontsize=9, color='white', fontweight='bold', zorder=103)
    artists.append(dis_text)
    
    return artists

# Create figure with team background
fig = plt.figure(figsize=(16, 10), facecolor=colors['background'])
ax = fig.add_subplot(111, facecolor=colors['field'])

ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
ax.set_aspect('equal')

draw_field(ax, colors)

# Initialize plot elements
offense_scatter = ax.scatter([], [], c=colors['offense_color'], s=350, 
                             edgecolors='white', linewidths=2.5, zorder=5, 
                             label='Offense', alpha=0.95)
defense_scatter = ax.scatter([], [], c=colors['defense_color'], s=350, 
                             edgecolors=colors['primary'], linewidths=2.5, 
                             zorder=5, label='Defense', alpha=0.95)

player_labels = []
stress_circles = []
gauge_artists = []

# Stats text
stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                     fontsize=14, verticalalignment='top',
                     color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=colors['primary'], 
                             alpha=0.85, edgecolor=colors['secondary'], linewidth=2))

# Title
title = ax.text(0.5, 0.98, 'Baltimore Ravens vs San Francisco 49ers - Christmas Day 2023', 
                transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', horizontalalignment='center',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=colors['primary'], 
                        alpha=0.85, edgecolor=colors['secondary'], linewidth=3))

ax.set_xlabel('Yards', color='white', fontsize=12, fontweight='bold')
ax.set_ylabel('Field Width (yards)', color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='white')

# Store previous frame data for DIS calculation
prev_defense_positions = None

def init():
    """Initialize animation"""
    offense_scatter.set_offsets(np.empty((0, 2)))
    defense_scatter.set_offsets(np.empty((0, 2)))
    stats_text.set_text('')
    return [offense_scatter, defense_scatter, stats_text]

def update(frame_num):
    """Update animation for each frame"""
    global prev_defense_positions
    
    frame_id = sorted(play_data['frame_id'].unique())[frame_num]
    current_frame = play_data[play_data['frame_id'] == frame_id]
    
    actual_positions = current_frame[~current_frame['is_projection']]
    
    # Clear previous elements
    for label in player_labels:
        label.remove()
    player_labels.clear()
    
    for circle in stress_circles:
        circle.remove()
    stress_circles.clear()
    
    for artist in gauge_artists:
        artist.remove()
    gauge_artists.clear()
    
    # Update player positions
    offense_data = actual_positions[actual_positions['player_side'] == 'Offense']
    defense_data = actual_positions[actual_positions['player_side'] == 'Defense']
    
    if len(offense_data) > 0:
        offense_scatter.set_offsets(offense_data[['x', 'y']].values)
    else:
        offense_scatter.set_offsets(np.empty((0, 2)))
    
    if len(defense_data) > 0:
        defense_scatter.set_offsets(defense_data[['x', 'y']].values)
    else:
        defense_scatter.set_offsets(np.empty((0, 2)))
    
    # Calculate or retrieve DCI/DIS
    if METRIC_MODE == 'precomputed' and has_dci and has_dis:
        # Use precomputed values (take mean if multiple players)
        dci_value = current_frame[DCI_COLUMN].mean()
        dis_value = current_frame[DIS_COLUMN].mean()
    else:
        # Calculate geometric proxies
        if len(defense_data) > 0:
            defense_positions = defense_data[['x', 'y']].values
            dci_value = calculate_geometric_dci(defense_positions)
            dis_value = calculate_geometric_dis(defense_positions, prev_defense_positions)
            prev_defense_positions = defense_positions
        else:
            dci_value = 0.5
            dis_value = 0.5
    
    # Draw DCI/DIS Gauge
    if SHOW_DCI_DIS_GAUGE:
        new_gauge_artists = draw_dci_dis_gauge(ax, dci_value, dis_value, colors)
        gauge_artists.extend(new_gauge_artists)
    
    # Add node stress halos
    if SHOW_NODE_STRESS and 'node_stress' in actual_positions.columns:
        for _, player in actual_positions.iterrows():
            if pd.notna(player['node_stress']):
                stress = player['node_stress']
                
                # Adjusted thresholds for better distribution (0.06-0.46 range for Seahawks)
                if stress < 0.20:  # Lower threshold for low stress
                    color = '#00FF80'  # Bright green for low stress
                elif stress < 0.35:  # Medium threshold
                    color = '#FFD700'  # Gold for medium stress  
                else:
                    color = '#FF4500'  # Orange-red for high stress
                
                # Larger radius and more visible alpha
                radius = 2.5 + (stress * 4.0)  # Even larger for visibility
                alpha = 0.3 + (stress * 0.4)   # More visible
                
                circle = Circle((player['x'], player['y']), radius, 
                              color=color, alpha=alpha, 
                              zorder=1, linewidth=3, edgecolor='white')
                ax.add_patch(circle)
                stress_circles.append(circle)
    
    # Add player labels
    if SHOW_POSITIONS or SHOW_PLAYER_NAMES or SHOW_PLAYER_NUMBERS:
        for _, player in actual_positions.iterrows():
            label_parts = []
            
            # Priority order: Position > Name > Number > ID
            if SHOW_POSITIONS and 'player_position' in actual_positions.columns and pd.notna(player['player_position']):
                label_parts.append(player['player_position'])
            elif SHOW_PLAYER_NAMES and int(player['nfl_id']) in PLAYER_NAMES:
                label_parts.append(PLAYER_NAMES[int(player['nfl_id'])])
            elif SHOW_PLAYER_NUMBERS and 'jersey_number' in actual_positions.columns and pd.notna(player['jersey_number']):
                label_parts.append(f"#{int(player['jersey_number'])}")
            else:
                label_parts.append(str(int(player['nfl_id'])))
            
            # Only show combined name+position if SHOW_POSITIONS is True
            if SHOW_PLAYER_NAMES and int(player['nfl_id']) in PLAYER_NAMES and SHOW_POSITIONS and 'player_position' in actual_positions.columns and pd.notna(player['player_position']):
                # Show both name and position
                label_text = f"{PLAYER_NAMES[int(player['nfl_id'])]}\n{player['player_position']}"
                fontsize = 8
            else:
                label_text = label_parts[0]
                fontsize = 9
            
            if player['player_side'] == 'Offense':
                bg_color = colors['offense_color']
                txt_color = colors['primary']
            else:
                bg_color = colors['defense_color']
                txt_color = colors['primary']
            
            label = ax.text(player['x'], player['y'] - 1.5, label_text,
                          ha='center', va='top', fontsize=fontsize, color=txt_color,
                          fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor=bg_color, alpha=0.85, 
                                  edgecolor='white', linewidth=1))
            player_labels.append(label)
    
    # Update stats
    offense_count = len(offense_data)
    defense_count = len(defense_data)
    num_frames = play_data['frame_id'].nunique()
    
    stats = f"Frame: {frame_num + 1}/{num_frames}\n"
    stats += f"OFF: {offense_count} | DEF: {defense_count}\n"
    
    if 'node_stress' in actual_positions.columns:
        avg_stress = actual_positions['node_stress'].mean()
        if pd.notna(avg_stress):
            stats += f"Avg Stress: {avg_stress:.3f}\n"
    
    stats += f"\nDCI: {dci_value:.3f}\nDIS: {dis_value:.3f}"
    
    # Add node stress legend if enabled
    if SHOW_NODE_STRESS:
        stats += f"\n\nSTRESS LEGEND:"
        stats += f"\nSafe (< 0.20)"  
        stats += f"\nAt Risk (0.20-0.35)"
        stats += f"\nBreaking (> 0.35)"
    
    stats_text.set_text(stats)
    
    elements = [offense_scatter, defense_scatter, stats_text]
    elements.extend(player_labels)
    elements.extend(stress_circles)
    elements.extend(gauge_artists)
    
    return elements

# Create animation
num_frames = play_data['frame_id'].nunique()
print(f"\nGenerating animation with {num_frames} frames...")

anim = FuncAnimation(fig, update, init_func=init, 
                    frames=num_frames, interval=100, blit=True)

# Generate output filename
output_base = f"{TEAM_THEME}_game{GAME_ID}_play{PLAY_ID}_metrics"

# Save animation
if VIDEO_FORMAT in ['mp4', 'both']:
    output_path_mp4 = f'{OUTPUT_DIR}/{output_base}.mp4'
    print(f"Saving MP4: {output_base}.mp4...")
    writer_mp4 = FFMpegWriter(fps=FPS, bitrate=2000)
    anim.save(output_path_mp4, writer=writer_mp4)
    print("✓ MP4 saved")

if VIDEO_FORMAT in ['gif', 'both']:
    output_path_gif = f'{OUTPUT_DIR}/{output_base}.gif'
    print(f"Saving GIF: {output_base}.gif...")
    writer_gif = PillowWriter(fps=FPS)
    anim.save(output_path_gif, writer=writer_gif)
    print("✓ GIF saved")

plt.close()

print(f"\n{'='*60}")
print(f"ANIMATION COMPLETE!")
print(f"{'='*60}")
print(f"Output: {OUTPUT_DIR}/{output_base}.{VIDEO_FORMAT}")
print(f"Metric Mode: {METRIC_MODE}")
print(f"\nDCI: Defensive Coverage Index (0=loose, 1=tight)")
print(f"DIS: Defensive Integrity Score (0=chaotic, 1=disciplined)")
