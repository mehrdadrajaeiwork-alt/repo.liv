import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide")

# --- API Key ---
API_KEY = "9e02904d18b54921bbf5128c1719ece6" # <<< API Token شما در اینجا قرار گرفت
API_URL = "https://api.football-data.org/v4/"
TEAM_ID = "64" # Liverpool FC ID

# --- Mock Data for New Season (Replace with actual API calls when available) ---
# Data for the 2026-27 season based on recent search results
mock_squad_data_2026_27 = {
    'name': [
        'Alisson Becker', 'Giorgi Mamardashvili', 'Freddie Woodman', 'Armin Pecsi', 'Harvey Davies',
        'Joe Gomez', 'Virgil van Dijk', 'Ibrahima Konate', 'Milos Kerkez', 'Conor Bradley',
        'Giovanni Leoni', 'Andy Robertson', 'Jeremie Frimpong', 'Rhys Williams', 'Calvin Ramsay',
        'Wataru Endo', 'Florian Wirtz', 'Dominik Szoboszlai', 'Alexis Mac Allister', 'Curtis Jones',
        'Ryan Gravenberch', 'Trey Nyoni', 'Stefan Bajcetic', 'James McConnell',
        'Alexander Isak', 'Mohamed Salah', 'Federico Chiesa', 'Cody Gakpo', 'Hugo Ekitike',
        'Rio Ngumoha', 'Diogo Jota'
    ],
    'position': [
        'Goalkeeper', 'Goalkeeper', 'Goalkeeper', 'Goalkeeper', 'Goalkeeper',
        'Defender', 'Defender', 'Defender', 'Defender', 'Defender',
        'Defender', 'Defender', 'Defender', 'Defender', 'Defender',
        'Midfielder', 'Midfielder', 'Midfielder', 'Midfielder', 'Midfielder',
        'Midfielder', 'Midfielder', 'Midfielder', 'Midfielder',
        'Forward', 'Forward', 'Forward', 'Forward', 'Forward',
        'Forward', 'Forward'
    ],
    'shirtNumber': [1, 25, 28, 41, 95, 2, 4, 5, 6, 12, 15, 26, 30, 46, 47, 3, 7, 8, 10, 17, 38, 42, 43, 53, 9, 11, 14, 18, 22, 73, 20],
}
df_squad_2026_27 = pd.DataFrame(mock_squad_data_2026_27)

# Mock data for recent matches (August-September 2026)
# Updated with results from previous searches and a placeholder for the next match
df_matches_2026_27 = pd.DataFrame({
    'date': ['2026-08-23', '2026-08-29', '2026-09-04', '2026-09-12'], # MW1, MW2, MW3, MW4
    'homeTeam': ['Newcastle United', 'Liverpool', 'Ipswich Town', 'Liverpool'],
    'awayTeam': ['Liverpool', 'Nottingham Forest', 'Liverpool', 'Fulham'],
    'score': ['2-2', '2-2', '0-2', None], # None for the upcoming match (Fulham)
    'fullTimeHome': [2, 2, 0, None],
    'fullTimeAway': [2, 2, 2, None],
    'competition': ['Premier League', 'Premier League', 'Premier League', 'Premier League']
})

# Mock data for League Table (after Matchweek 3)
# Based on the search result: https://en.wikipedia.org/wiki/2026%E2%80%9327_Premier_League
df_league_table_2026_27 = pd.DataFrame({
    'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Team': [
        'Manchester City', 'Arsenal', 'Hull City', 'Chelsea', 'Brentford', 'Liverpool',
        'Newcastle United', 'Everton', 'Leeds United', 'Brighton & Hove Albion',
        'Manchester United', 'Sunderland', 'Crystal Palace', 'Ipswich Town', 'Bournemouth',
        'Nottingham Forest', 'Aston Villa', 'Tottenham Hotspur', 'Fulham', 'Coventry City'
    ],
    'Played': [3] * 20, # Assuming all played 3 matches
    'Won': [3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # Example data, Needs actual figures
    'Drawn': [0, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 3], # Example data
    'Lost': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Example data
    'GF': [9, 7, 5, 4, 5, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0], # Example data
    'GA': [1, 2, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3], # Example data
    'GD': [8, 5, 4, 2, 3, 2, 1, 0, 0, 0, -2, -2, -2, -2, -2, -1, -2, -3, -3, -3], # Example data
    'Points': [9, 9, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2, 2, 0] # Example data
})


# --- Helper Functions ---
# Function to fetch data from the API (simplified for this example)
@st.cache_data
def fetch_api_data(endpoint: str, params=None) -> pd.DataFrame | None:
    """Fetches data from the football-data.org API."""
    headers = {"X-Auth-Token": API_KEY}
    url = f"{API_URL}{endpoint}"
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        # Adapt based on the actual structure of API responses for different endpoints
        if 'matches' in data:
            return pd.DataFrame(data['matches'])
        if 'squad' in data:
            return pd.DataFrame(data['squad'])
        if 'standings' in data and data['standings']:
            # Process standings data if available and in the expected format
            processed_standings = []
            for standing in data['standings']:
                for team_standing in standing['table']:
                    processed_standings.append(team_standing)
            return pd.DataFrame(processed_standings)
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function to get player data (uses mock data for 2026-27)
def get_player_data_2026_27(player_name: str) -> pd.Series | None:
    """Retrieves mock squad data for a given player for the 2026-27 season."""
    player_row = df_squad_2026_27[df_squad_2026_27['name'].str.lower() == player_name.lower()]
    if not player_row.empty:
        return player_row.iloc[0]
    return None

# --- Tabbed Interface ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Main Dashboard", "Player Analysis", "Match History", "League Table", "Smart Prediction"])

with tab1:
    st.header("Liverpool FC - Season Dashboard 2026-27")
    st.write("Overview of Liverpool's performance in the current season.")

    # Displaying recent match results from mock data
    st.subheader("Recent Matches")
    recent_matches = df_matches_2026_27.head(3)
    st.dataframe(recent_matches[['date', 'homeTeam', 'awayTeam', 'score', 'competition']])

    # Placeholder for league standing (will be populated from tab4)
    st.subheader("Current League Standing")
    liverpool_standing = df_league_table_2026_27[df_league_table_2026_27['Team'] == 'Liverpool']
    if not liverpool_standing.empty:
        st.write(f"**Liverpool is currently:** Rank {liverpool_standing['Rank'].iloc[0]} with {liverpool_standing['Points'].iloc[0]} points.")
    else:
        st.write("League table data not available yet.")

with tab2:
    st.header("Player Analysis")
    st.write("Analyze individual player performance.")

    st.subheader("Liverpool Squad 2026-27")
    st.dataframe(df_squad_2026_27)

    st.subheader("Individual Player Stats")
    player_name_input = st.text_input("Enter player name to see details (e.g., Mohamed Salah):", key="player_analysis_input")

    if player_name_input:
        player_data = get_player_data_2026_27(player_name_input)
        if player_data is not None:
            st.write(f"**{player_name_input}**")
            st.write(f"**Position:** {player_data.get('position', 'N/A')}")
            st.write(f"**Shirt Number:** {player_data.get('shirtNumber', 'N/A')}")
        else:
            st.warning(f"Player '{player_name_input}' not found in the 2026-27 squad data.")
            st.info("Please ensure the player name is spelled correctly and is part of the current Liverpool squad.")

with tab3:
    st.header("Match History")
    st.write("Detailed history of Liverpool's matches.")

    st.subheader("Season 2026-27 Match Log")
    # Display all available match history from mock data
    st.dataframe(df_matches_2026_27[['date', 'homeTeam', 'awayTeam', 'score', 'competition']])

with tab4:
    st.header("League Table")
    st.write("Current Premier League Standings 2026-27")

    st.dataframe(df_league_table_2026_27.set_index('Rank'))

with tab5:
    st.header("Smart Prediction")
    st.write("Predicting future match outcomes.")

    st.subheader("Match Outcome Prediction (Placeholder)")
    st.write("Based on recent performance and general team stats.")

    # Simple prediction using mock data
    if not df_matches_2026_27.dropna(subset=['score']).empty:
        liverpool_goals_scored = []
        liverpool_goals_conceded = []
        opponent_goals_scored = []
        opponent_goals_conceded = []

        for index, row in df_matches_2026_27.dropna(subset=['score']).iterrows():
            home_team = row['homeTeam']
            away_team = row['awayTeam']
            score = row['score'].split('-')
            home_score = int(score[0])
            away_score = int(score[1])

            if home_team == 'Liverpool':
                liverpool_goals_scored.append(home_score)
                liverpool_goals_conceded.append(away_score)
                opponent_goals_scored.append(away_score)
                opponent_goals_conceded.append(home_score)
            elif away_team == 'Liverpool':
                liverpool_goals_scored.append(away_score)
                liverpool_goals_conceded.append(home_score)
                opponent_goals_scored.append(home_score)
                opponent_goals_conceded.append(away_score)

        if len(liverpool_goals_scored) >= 2:
            # Prepare data for regression model
            # Features: Average Liverpool Goals Scored, Average Liverpool Goals Conceded
            # Target: Probability of Liverpool Win (binary: 1 for win, 0 for not win)
            # This is a highly simplified approach. A real model would need more features and possibly different targets.
            
            # Using simplified features: average goals scored and conceded in past games
            X_features = []
            y_target = [] # 1 for win, 0 for draw/loss

            for i in range(len(liverpool_goals_scored)):
                # Simple moving average for features for predicting the *next* game
                avg_goals_scored = np.mean(liverpool_goals_scored[:i+1])
                avg_goals_conceded = np.mean(liverpool_goals_conceded[:i+1])
                X_features.append([avg_goals_scored, avg_goals_conceded])

                # Determine outcome for the game that generated these averages
                if liverpool_goals_scored[i] > opponent_goals_conceded[i]:
                    y_target.append(1) # Liverpool won
                else:
                    y_target.append(0) # Liverpool drew or lost

            if len(X_features) >= 2: # Need at least 2 data points to train a basic model
                X = np.array(X_features)
                y = np.array(y_target)

                model = LinearRegression()
                model.fit(X, y)

                # Predict for the next match (vs Fulham)
                # Using the latest available average stats for Liverpool
                latest_avg_goals_scored = np.mean(liverpool_goals_scored) if liverpool_goals_scored else 1.8
                latest_avg_goals_conceded = np.mean(liverpool_goals_conceded) if liverpool_goals_conceded else 1.2

                prediction_input = np.array([[latest_avg_goals_scored, latest_avg_goals_conceded]])
                win_probability = model.predict(prediction_input)[0]
                win_probability = np.clip(win_probability, 0, 1) # Clip probability

                st.write(f"**Prediction for next match vs Fulham:**")
                st.write(f"- Probability of Liverpool Win: {win_probability:.2%}")
                # Further enhancements could include predicting draws/losses or scorelines
            else:
                st.info("Not enough historical match data to train the prediction model.")
        else:
            st.info("Not enough historical match data to make a prediction.")

    else:
        st.info("No completed matches found to base predictions on.")
