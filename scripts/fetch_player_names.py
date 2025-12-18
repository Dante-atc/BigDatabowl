#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 12: PLAYER NAME & ROSTER FETCHER
======================================

SUMMARY
-------
This utility script resolves NFL player names for animation and visualization.
It bridges the gap between anonymous tracking data IDs and real player names by
fetching roster data from external APIs (ESPN) or using manual fallbacks.

METHODOLOGY
-----------
1. Game Identification: Determines the home and away teams based on the game date.
2. Roster Scraping: Queries ESPN's public API to fetch active rosters for the
   identified teams (2023 Season).
3. Fuzzy Matching: Maps tracking data positions (e.g., 'ILB', 'FS') to official
   roster positions using a heuristic matching algorithm.
4. Fallback Handling: Includes manual roster dictionaries for specific games
   (e.g., Vikings vs. 49ers) in case API calls fail or time out.

INPUTS
------
1. vikings_game_full.csv (Tracking data with 'nfl_id' and 'player_position')

OUTPUTS
-------
1. player_names_lookup.json (Key-Value mapping of NFL_ID -> Player Name)
"""

import pandas as pd
import requests
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_FILE = 'C:\\Users\\Usuario\\Documents\\devops\\d\\BigDataBowl\\vikings_game_full.csv'
OUTPUT_FILE = 'player_names_lookup.json'

print(f"\n{'='*60}")
print(f"NFL PLAYER NAME FETCHER")
print(f"{'='*60}\n")

# Load data and analyze
df = pd.read_csv(CSV_FILE)

# Extract game information
game_id = df['game_id'].iloc[0]
game_date = str(game_id)[:8]  # 20231023
formatted_date = f"{game_date[:4]}-{game_date[4:6]}-{game_date[6:8]}"  # 2023-10-23

print(f"Game ID: {game_id}")
print(f"Game Date: {formatted_date}")

# Get unique players
unique_players = df.groupby('nfl_id').agg({
    'player_position': 'first',
    'player_side': 'first'
}).reset_index()

print(f"Total unique players: {len(unique_players)}")
print(f"Offense players: {len(unique_players[unique_players['player_side'] == 'Offense'])}")
print(f"Defense players: {len(unique_players[unique_players['player_side'] == 'Defense'])}")

# ============================================================================
# NFL TEAM MAPPING (2023 Season)
# ============================================================================

def get_team_from_game_date():
    """
    Determine teams based on game date and known schedules
    October 23, 2023 was Week 7 of NFL season
    """
    
    # This would be Minnesota Vikings game based on the filename
    # October 23, 2023 - Vikings @ 49ers (Monday Night Football)
    
    teams = {
        'home_team': 'SF',  # San Francisco 49ers  
        'away_team': 'MIN'  # Minnesota Vikings
    }
    
    return teams

def fetch_nfl_roster_espn(team_code, year=2023):
    """
    Fetch NFL roster from ESPN API
    """
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_code}/roster"
        
        print(f"  Fetching {team_code} roster from ESPN...")
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            players = {}
            
            if 'athletes' in data:
                for group in data['athletes']:
                    if 'items' in group:
                        for player in group['items']:
                            player_name = player.get('fullName', '')
                            position = player.get('position', {}).get('abbreviation', '')
                            
                            # Try to get jersey number
                            jersey = player.get('jersey', '')
                            
                            players[player_name] = {
                                'position': position,
                                'jersey': jersey,
                                'team': team_code
                            }
            
            print(f"    Found {len(players)} players for {team_code}")
            return players
            
        else:
            print(f"    Failed to fetch {team_code} roster (HTTP {response.status_code})")
            return {}
            
    except Exception as e:
        print(f"    Error fetching {team_code} roster: {str(e)}")
        return {}

def fetch_nfl_roster_api(team_code, year=2023):
    """
    Alternative: Try to fetch from NFL.com API or other sources
    """
    try:
        # NFL.com API endpoint (may require different formatting)
        url = f"https://www.nfl.com/teams/{team_code.lower()}/roster/"
        
        print(f"  Attempting alternative source for {team_code}...")
        
        # This is a placeholder - would need to implement web scraping
        # or find another public API
        
        return {}
        
    except Exception as e:
        print(f"    Error with alternative source: {str(e)}")
        return {}

def match_players_to_roster(unique_players, all_rosters):
    """
    Match players in our data to roster players based on position
    """
    
    player_matches = {}
    
    print(f"\nMatching players to rosters...")
    
    for _, player_data in unique_players.iterrows():
        nfl_id = player_data['nfl_id']
        position = player_data['player_position']
        side = player_data['player_side']
        
        print(f"  Looking for {side} {position} (ID: {nfl_id})...")
        
        best_matches = []
        
        # Search through all roster players
        for player_name, roster_info in all_rosters.items():
            roster_pos = roster_info.get('position', '')
            
            # Position matching logic
            position_match = False
            
            if position == roster_pos:
                position_match = True
            elif position == 'ILB' and roster_pos in ['LB', 'MLB']:
                position_match = True
            elif position == 'OLB' and roster_pos in ['LB', 'OLB']:
                position_match = True
            elif position == 'MLB' and roster_pos in ['LB', 'ILB', 'MLB']:
                position_match = True
                
            if position_match:
                best_matches.append({
                    'name': player_name,
                    'team': roster_info.get('team'),
                    'jersey': roster_info.get('jersey'),
                    'position': roster_pos
                })
        
        if best_matches:
            if len(best_matches) == 1:
                match = best_matches[0]
                player_matches[nfl_id] = match['name']
                print(f"    ✓ Found: {match['name']} ({match['team']} {match['position']})")
            else:
                # Multiple matches - take first one
                match = best_matches[0]
                player_matches[nfl_id] = match['name']
                print(f"    ? Multiple matches, using: {match['name']} ({len(best_matches)} options)")
        else:
            # No match found
            player_matches[nfl_id] = f"{side}_{position}"
            print(f"    ✗ No match found, using generic name")
    
    return player_matches

def create_manual_vikings_roster():
    """
    Create a manual roster for Vikings key players (2023 season)
    This is a fallback if APIs don't work
    """
    
    vikings_2023 = {
        # Offense
        'Kirk Cousins': {'position': 'QB', 'jersey': '8', 'team': 'MIN'},
        'Alexander Mattison': {'position': 'RB', 'jersey': '2', 'team': 'MIN'},
        'Dalvin Cook': {'position': 'RB', 'jersey': '4', 'team': 'MIN'},
        'Justin Jefferson': {'position': 'WR', 'jersey': '18', 'team': 'MIN'},
        'Adam Thielen': {'position': 'WR', 'jersey': '19', 'team': 'MIN'},
        'Jordan Addison': {'position': 'WR', 'jersey': '3', 'team': 'MIN'},
        'T.J. Hockenson': {'position': 'TE', 'jersey': '87', 'team': 'MIN'},
        'Kyle Rudolph': {'position': 'TE', 'jersey': '82', 'team': 'MIN'},
        
        # Defense  
        'Harrison Smith': {'position': 'SS', 'jersey': '22', 'team': 'MIN'},
        'Camryn Bynum': {'position': 'FS', 'jersey': '24', 'team': 'MIN'},
        'Byron Murphy Jr.': {'position': 'CB', 'jersey': '7', 'team': 'MIN'},
        'Akayleb Evans': {'position': 'CB', 'jersey': '21', 'team': 'MIN'},
        'Jordan Hicks': {'position': 'LB', 'jersey': '58', 'team': 'MIN'},
        'Brian Asamoah': {'position': 'LB', 'jersey': '54', 'team': 'MIN'},
        'Danielle Hunter': {'position': 'OLB', 'jersey': '99', 'team': 'MIN'},
        'Za\'Darius Smith': {'position': 'OLB', 'jersey': '55', 'team': 'MIN'},
    }
    
    return vikings_2023

def create_manual_49ers_roster():
    """
    Create a manual roster for 49ers key players (2023 season)
    """
    
    niners_2023 = {
        # Offense
        'Brock Purdy': {'position': 'QB', 'jersey': '13', 'team': 'SF'},
        'Christian McCaffrey': {'position': 'RB', 'jersey': '23', 'team': 'SF'},
        'Jordan Mason': {'position': 'RB', 'jersey': '24', 'team': 'SF'},
        'Deebo Samuel': {'position': 'WR', 'jersey': '1', 'team': 'SF'},
        'Brandon Aiyuk': {'position': 'WR', 'jersey': '11', 'team': 'SF'},
        'George Kittle': {'position': 'TE', 'jersey': '85', 'team': 'SF'},
        
        # Defense
        'Talanoa Hufanga': {'position': 'SS', 'jersey': '29', 'team': 'SF'},
        'Ji\'Ayir Brown': {'position': 'FS', 'jersey': '27', 'team': 'SF'},
        'Charvarius Ward': {'position': 'CB', 'jersey': '7', 'team': 'SF'},
        'Deommodore Lenoir': {'position': 'CB', 'jersey': '38', 'team': 'SF'},
        'Fred Warner': {'position': 'LB', 'jersey': '54', 'team': 'SF'},
        'Dre Greenlaw': {'position': 'LB', 'jersey': '57', 'team': 'SF'},
        'Nick Bosa': {'position': 'OLB', 'jersey': '97', 'team': 'SF'},
        'Leonard Floyd': {'position': 'OLB', 'jersey': '56', 'team': 'SF'},
    }
    
    return niners_2023

# ============================================================================
# MAIN EXECUTION
# ============================================================================

try:
    # Determine teams
    teams = get_team_from_game_date()
    print(f"\nTeams: {teams['away_team']} @ {teams['home_team']}")
    
    # Try to fetch rosters from APIs
    all_rosters = {}
    
    print(f"\nAttempting to fetch rosters from online sources...")
    
    # ESPN team codes mapping
    espn_codes = {
        'MIN': '16',  # Minnesota Vikings
        'SF': '25'    # San Francisco 49ers  
    }
    
    for team, espn_code in espn_codes.items():
        if team in [teams['home_team'], teams['away_team']]:
            roster = fetch_nfl_roster_espn(espn_code)
            all_rosters.update(roster)
            time.sleep(1)  # Be respectful to APIs
    
    # If API fetching failed, use manual rosters
    if len(all_rosters) == 0:
        print(f"\nAPI fetch failed, using manual rosters...")
        
        vikings_roster = create_manual_vikings_roster()
        niners_roster = create_manual_49ers_roster()
        
        all_rosters.update(vikings_roster)
        all_rosters.update(niners_roster)
        
        print(f"  Loaded manual rosters: {len(all_rosters)} players")
    
    # Match players
    player_matches = match_players_to_roster(unique_players, all_rosters)
    
    # Save results
    print(f"\nSaving player lookup to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(player_matches, f, indent=2)
    
    print(f"✓ Saved {len(player_matches)} player name mappings")
    
    # Display results
    print(f"\nPlayer Name Mappings:")
    print(f"{'='*50}")
    
    for idx, (nfl_id, player_name) in enumerate(sorted(player_matches.items())):
        player_info = unique_players[unique_players['nfl_id'] == int(nfl_id)]
        if len(player_info) > 0:
            position = player_info.iloc[0]['player_position']
            side = player_info.iloc[0]['player_side']
            print(f"  {nfl_id}: {player_name} ({side} {position})")
    
    # Create code snippet for animation script
    print(f"\nCode snippet for animation3.py:")
    print(f"{'='*50}")
    print(f"PLAYER_NAMES = {{")
    for nfl_id, player_name in sorted(player_matches.items()):
        # Clean up name for code
        clean_name = player_name.replace("'", "\\'")
        print(f"    {nfl_id}: '{clean_name}',")
    print(f"}}")

except Exception as e:
    print(f"\nError in player fetching: {str(e)}")
    print(f"\nFalling back to manual roster creation...")
    
    # Create basic mappings as fallback
    fallback_matches = {}
    for idx, row in unique_players.iterrows():
        nfl_id = row['nfl_id']
        position = row['player_position']
        side = row['player_side']
        fallback_matches[str(nfl_id)] = f"{side}_{position}_{nfl_id}"
    
    with open('fallback_player_names.json', 'w') as f:
        json.dump(fallback_matches, f, indent=2)
    
    print(f"Created fallback mapping with {len(fallback_matches)} entries")

print(f"\n{'='*60}")
print(f"PLAYER NAME FETCHING COMPLETE!")
print(f"{'='*60}")
