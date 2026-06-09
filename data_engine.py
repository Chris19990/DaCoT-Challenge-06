"""
data_engine.py — Génère les données synthétiques fidèles au notebook World Cup
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 1. DONNÉES WORLDCUPS (éditions)
# ─────────────────────────────────────────────
WORLDCUPS_RAW = [
    (1930,"Uruguay",13,18,434500,70000),
    (1934,"Italy",16,17,395000,50000),
    (1938,"France",15,18,483000,45000),
    (1950,"Brazil",13,22,1337000,68000),
    (1954,"Switzerland",16,26,943000,60000),
    (1958,"Sweden",16,35,868000,52000),
    (1962,"Chile",16,32,776000,68000),
    (1966,"England",16,32,1614677,98000),
    (1970,"Mexico",16,32,1673975,107000),
    (1974,"Germany",16,38,1865762,62000),
    (1978,"Argentina",16,38,1610215,77000),
    (1982,"Spain",24,52,2109723,90000),
    (1986,"Mexico",24,52,2394031,114600),
    (1990,"Italy",24,52,2516215,73603),
    (1994,"USA",24,52,3587538,93869),
    (1998,"France",32,64,2785100,80000),
    (2002,"Japan/Korea",32,64,2705197,69029),
    (2006,"Germany",32,64,3359439,66000),
    (2010,"South Africa",32,64,3178856,94700),
    (2014,"Brazil",32,64,3429873,74738),
]

# ─────────────────────────────────────────────
# 2. DONNÉES MATCHS (852 matchs 1930-2014)
# ─────────────────────────────────────────────
MATCHES_RAW = [
    # 1930
    (1930,"France","Mexico",4,1,"Group 1"),(1930,"USA","Belgium",3,0,"Group 4"),
    (1930,"Yugoslavia","Brazil",2,1,"Group 2"),(1930,"Romania","Peru",3,1,"Group 3"),
    (1930,"Argentina","France",1,0,"Group 1"),(1930,"Chile","Mexico",3,0,"Group 1"),
    (1930,"Yugoslavia","Bolivia",4,0,"Group 2"),(1930,"USA","Paraguay",3,0,"Group 4"),
    (1930,"Uruguay","Peru",1,0,"Group 3"),(1930,"Chile","France",1,0,"Group 1"),
    (1930,"Argentina","Mexico",6,3,"Group 1"),(1930,"Brazil","Bolivia",4,0,"Group 2"),
    (1930,"Uruguay","Romania",4,0,"Group 3"),(1930,"Argentina","Chile",3,1,"Group 1"),
    (1930,"USA","Belgium",3,0,"Group 4"),(1930,"Argentina","USA",6,1,"Semi-Final"),
    (1930,"Uruguay","Yugoslavia",6,1,"Semi-Final"),(1930,"Uruguay","Argentina",4,2,"Final"),
    # 1934
    (1934,"Italy","USA",7,1,"First Round"),(1934,"Czechoslovakia","Romania",2,1,"First Round"),
    (1934,"Germany","Belgium",5,2,"First Round"),(1934,"Austria","France",3,2,"First Round"),
    (1934,"Spain","Brazil",3,1,"First Round"),(1934,"Hungary","Egypt",4,2,"First Round"),
    (1934,"Sweden","Argentina",3,2,"First Round"),(1934,"Switzerland","Netherlands",3,2,"First Round"),
    (1934,"Germany","Sweden",2,1,"Quarter-Final"),(1934,"Austria","Hungary",2,1,"Quarter-Final"),
    (1934,"Italy","Spain",1,1,"Quarter-Final"),(1934,"Czechoslovakia","Switzerland",3,2,"Quarter-Final"),
    (1934,"Italy","Spain",1,0,"Quarter-Final replay"),(1934,"Germany","Austria",3,2,"Semi-Final"),
    (1934,"Italy","Austria",1,0,"Semi-Final"),(1934,"Czechoslovakia","Germany",3,1,"Third place"),
    (1934,"Italy","Czechoslovakia",2,1,"Final"),
    # 1938
    (1938,"Switzerland","Germany",1,1,"First Round"),(1938,"Cuba","Romania",3,3,"First Round"),
    (1938,"Hungary","Dutch East Indies",6,0,"First Round"),(1938,"France","Belgium",3,1,"First Round"),
    (1938,"Czechoslovakia","Netherlands",3,0,"First Round"),(1938,"Brazil","Poland",6,5,"First Round"),
    (1938,"Italy","Norway",2,1,"First Round"),(1938,"Switzerland","Germany",4,2,"First Round replay"),
    (1938,"Cuba","Romania",2,1,"First Round replay"),(1938,"Hungary","Switzerland",2,0,"Quarter-Final"),
    (1938,"Italy","France",3,1,"Quarter-Final"),(1938,"Brazil","Czechoslovakia",1,1,"Quarter-Final"),
    (1938,"Cuba","Sweden",0,8,"Quarter-Final"),(1938,"Brazil","Czechoslovakia",2,1,"Quarter-Final replay"),
    (1938,"Italy","Brazil",2,1,"Semi-Final"),(1938,"Hungary","Sweden",5,1,"Semi-Final"),
    (1938,"Brazil","Sweden",4,2,"Third place"),(1938,"Italy","Hungary",4,2,"Final"),
    # 1950
    (1950,"Brazil","Mexico",4,0,"Group 1"),(1950,"Yugoslavia","Switzerland",3,0,"Group 2"),
    (1950,"Sweden","Italy",3,2,"Group 3"),(1950,"Uruguay","Bolivia",8,0,"Group 4"),
    (1950,"Brazil","Switzerland",2,2,"Group 1"),(1950,"Yugoslavia","Mexico",4,1,"Group 2"),
    (1950,"Spain","USA",3,1,"Group 2"),(1950,"Brazil","Yugoslavia",2,0,"Group 1"),
    (1950,"Sweden","Paraguay",2,2,"Group 3"),(1950,"Italy","Paraguay",2,0,"Group 3"),
    (1950,"Switzerland","Mexico",2,1,"Group 1"),(1950,"Spain","Chile",2,0,"Group 2"),
    (1950,"Uruguay","Spain",2,2,"Final Round"),(1950,"Brazil","Sweden",7,1,"Final Round"),
    (1950,"Uruguay","Sweden",3,2,"Final Round"),(1950,"Brazil","Spain",6,1,"Final Round"),
    (1950,"Uruguay","Brazil",2,1,"Final Round"),(1950,"Spain","Sweden",1,3,"Final Round"),
    # 1954
    (1954,"Yugoslavia","France",1,0,"Group 1"),(1954,"Brazil","Mexico",5,0,"Group 1"),
    (1954,"Hungary","Germany",8,3,"Group 2"),(1954,"Turkey","Germany",4,1,"Group 2"),
    (1954,"Hungary","Germany",8,3,"Group 2"),(1954,"Uruguay","Czechoslovakia",2,0,"Group 3"),
    (1954,"Austria","Scotland",5,0,"Group 4"),(1954,"Uruguay","Scotland",7,0,"Group 3"),
    (1954,"Brazil","Yugoslavia",1,1,"Group 1"),(1954,"Hungary","Brazil",4,2,"Quarter-Final"),
    (1954,"Austria","Switzerland",7,5,"Quarter-Final"),(1954,"Uruguay","England",4,2,"Quarter-Final"),
    (1954,"Germany","Yugoslavia",2,0,"Quarter-Final"),(1954,"Germany","Austria",6,1,"Semi-Final"),
    (1954,"Hungary","Uruguay",4,2,"Semi-Final"),(1954,"Austria","Uruguay",3,1,"Third place"),
    (1954,"Germany","Hungary",3,2,"Final"),
    # 1958
    (1958,"Germany","Argentina",3,1,"Group 1"),(1958,"France","Paraguay",7,3,"Group 1"),
    (1958,"Yugoslavia","Scotland",1,1,"Group 2"),(1958,"France","Yugoslavia",3,2,"Group 1"),
    (1958,"Germany","Czechoslovakia",2,2,"Group 1"),(1958,"Argentina","Northern Ireland",3,1,"Group 1"),
    (1958,"Brazil","Austria",3,0,"Group 4"),(1958,"Sweden","Mexico",3,0,"Group 3"),
    (1958,"Hungary","Wales",1,1,"Group 3"),(1958,"Brazil","England",0,0,"Group 4"),
    (1958,"Sweden","Hungary",2,1,"Group 3"),(1958,"Wales","Hungary",2,1,"Quarter-Final playoff"),
    (1958,"France","Scotland",2,1,"Group 1"),(1958,"Brazil","USSR",2,0,"Quarter-Final"),
    (1958,"Sweden","USSR",2,0,"Quarter-Final"),(1958,"France","Northern Ireland",4,0,"Quarter-Final"),
    (1958,"Germany","Yugoslavia",1,0,"Quarter-Final"),(1958,"Wales","Brazil",0,1,"Quarter-Final"),
    (1958,"Brazil","France",5,2,"Semi-Final"),(1958,"Sweden","Germany",3,1,"Semi-Final"),
    (1958,"France","Germany",6,3,"Third place"),(1958,"Brazil","Sweden",5,2,"Final"),
    # 1962
    (1962,"Uruguay","Colombia",2,1,"Group 1"),(1962,"USSR","Yugoslavia",2,0,"Group 2"),
    (1962,"Chile","Switzerland",3,1,"Group 2"),(1962,"Germany","Italy",0,0,"Group 2"),
    (1962,"Chile","Italy",2,0,"Group 2"),(1962,"Brazil","Mexico",2,0,"Group 3"),
    (1962,"Hungary","England",2,1,"Group 4"),(1962,"Brazil","Czechoslovakia",0,0,"Group 3"),
    (1962,"Argentina","Bulgaria",1,0,"Group 4"),(1962,"Brazil","Spain",2,1,"Group 3"),
    (1962,"Chile","USSR",2,1,"Quarter-Final"),(1962,"Yugoslavia","Germany",1,0,"Quarter-Final"),
    (1962,"Brazil","England",3,1,"Quarter-Final"),(1962,"Czechoslovakia","Hungary",1,0,"Quarter-Final"),
    (1962,"Brazil","Chile",4,2,"Semi-Final"),(1962,"Czechoslovakia","Yugoslavia",3,1,"Semi-Final"),
    (1962,"Chile","Yugoslavia",1,0,"Third place"),(1962,"Brazil","Czechoslovakia",3,1,"Final"),
    # 1966
    (1966,"England","Uruguay",0,0,"Group 1"),(1966,"France","Mexico",1,1,"Group 1"),
    (1966,"Portugal","Hungary",3,1,"Group 3"),(1966,"USSR","North Korea",3,0,"Group 4"),
    (1966,"Brazil","Bulgaria",2,0,"Group 3"),(1966,"Germany","Switzerland",5,0,"Group 2"),
    (1966,"Argentina","Spain",2,1,"Group 2"),(1966,"England","Mexico",2,0,"Group 1"),
    (1966,"Hungary","Brazil",3,1,"Group 3"),(1966,"Portugal","Bulgaria",3,0,"Group 3"),
    (1966,"Germany","Argentina",0,0,"Group 2"),(1966,"USSR","Chile",2,1,"Group 4"),
    (1966,"England","France",2,0,"Group 1"),(1966,"Portugal","Brazil",3,1,"Group 3"),
    (1966,"Germany","Spain",2,1,"Group 2"),(1966,"Argentina","Germany",0,0,"Quarter-Final"),
    (1966,"England","Argentina",1,0,"Quarter-Final"),(1966,"Portugal","North Korea",5,3,"Quarter-Final"),
    (1966,"USSR","Hungary",2,1,"Quarter-Final"),(1966,"England","Portugal",2,1,"Semi-Final"),
    (1966,"Germany","USSR",2,1,"Semi-Final"),(1966,"Portugal","USSR",2,1,"Third place"),
    (1966,"England","Germany",4,2,"Final"),
    # 1970
    (1970,"Mexico","USSR",0,0,"Group 1"),(1970,"Belgium","El Salvador",3,0,"Group 1"),
    (1970,"Italy","Sweden",1,0,"Group 2"),(1970,"Brazil","Czechoslovakia",4,1,"Group 3"),
    (1970,"Germany","Morocco",2,1,"Group 4"),(1970,"Mexico","Belgium",4,1,"Group 1"),
    (1970,"USSR","Belgium",4,1,"Group 1"),(1970,"Italy","Uruguay",0,0,"Group 2"),
    (1970,"Brazil","England",1,0,"Group 3"),(1970,"Germany","Bulgaria",5,2,"Group 4"),
    (1970,"Brazil","Romania",3,2,"Group 3"),(1970,"Germany","Peru",3,1,"Group 4"),
    (1970,"Uruguay","USSR",1,0,"Quarter-Final"),(1970,"Italy","Mexico",4,1,"Quarter-Final"),
    (1970,"Brazil","Peru",4,2,"Quarter-Final"),(1970,"Germany","England",3,2,"Quarter-Final"),
    (1970,"Italy","Germany",4,3,"Semi-Final"),(1970,"Brazil","Uruguay",3,1,"Semi-Final"),
    (1970,"Germany","Uruguay",1,0,"Third place"),(1970,"Brazil","Italy",4,1,"Final"),
    # 1974
    (1974,"Brazil","Yugoslavia",0,0,"Group 1"),(1974,"Germany","Chile",1,0,"Group 1"),
    (1974,"Uruguay","Netherlands",0,2,"Group 3"),(1974,"Netherlands","Uruguay",2,0,"Group 3"),
    (1974,"Brazil","Scotland",0,0,"Group 1"),(1974,"Germany","Australia",3,0,"Group 1"),
    (1974,"Argentina","Italy",1,1,"Group 4"),(1974,"Netherlands","Bulgaria",4,1,"Group 3"),
    (1974,"Brazil","Zaire",3,0,"Group 2"),(1974,"Germany","East Germany",0,1,"Group 1"),
    (1974,"Netherlands","Argentina",4,0,"Group 4 2nd"),(1974,"Brazil","East Germany",1,0,"Group 2 2nd"),
    (1974,"Germany","Yugoslavia",2,0,"Group 1 2nd"),(1974,"Brazil","Argentina",2,1,"Group 2 2nd"),
    (1974,"Netherlands","Brazil",2,0,"Semi-Final"),(1974,"Germany","Poland",1,0,"Semi-Final"),
    (1974,"Poland","Brazil",1,0,"Third place"),(1974,"Germany","Netherlands",2,1,"Final"),
    # 1978
    (1978,"Argentina","Hungary",2,1,"Group 1"),(1978,"Italy","France",2,1,"Group 1"),
    (1978,"Germany","Poland",0,0,"Group 2"),(1978,"Tunisia","Mexico",3,1,"Group 2"),
    (1978,"Austria","Spain",2,1,"Group 3"),(1978,"Netherlands","Iran",3,0,"Group 4"),
    (1978,"Argentina","France",2,1,"Group 1"),(1978,"Italy","Hungary",3,1,"Group 1"),
    (1978,"Netherlands","Peru",4,1,"Group 4 2nd"),(1978,"Argentina","Poland",2,0,"Group 1 2nd"),
    (1978,"Italy","Germany",0,0,"Group 1 2nd"),(1978,"Netherlands","Italy",2,1,"Semi-Final"),
    (1978,"Argentina","Brazil",0,0,"Semi-Final"),(1978,"Brazil","Italy",2,1,"Third place"),
    (1978,"Argentina","Netherlands",3,1,"Final"),
    # 1982
    (1982,"Italy","Poland",0,0,"Group 1"),(1982,"Germany","Algeria",1,2,"Group 2"),
    (1982,"Brazil","USSR",2,1,"Group 6"),(1982,"Argentina","Belgium",0,1,"Group 3"),
    (1982,"England","France",3,1,"Group 4"),(1982,"Spain","Honduras",1,1,"Group 5"),
    (1982,"Italy","Cameroon",1,1,"Group 1"),(1982,"Brazil","Scotland",4,1,"Group 6"),
    (1982,"Germany","Chile",4,1,"Group 2"),(1982,"Argentina","Hungary",4,1,"Group 3"),
    (1982,"Brazil","Argentina",3,1,"Group 3 2nd"),(1982,"Italy","Argentina",2,1,"Group 1 2nd"),
    (1982,"Germany","England",0,0,"Group 2 2nd"),(1982,"France","Austria",1,0,"Group 4 2nd"),
    (1982,"Brazil","Italy",2,3,"Semi-Final"),(1982,"Germany","France",3,3,"Semi-Final"),
    (1982,"Poland","France",3,2,"Third place"),(1982,"Italy","Germany",3,1,"Final"),
    # 1986
    (1986,"Bulgaria","Italy",1,1,"Group A"),(1986,"Argentina","South Korea",3,1,"Group A"),
    (1986,"Mexico","Belgium",2,1,"Group B"),(1986,"Brazil","Spain",1,0,"Group D"),
    (1986,"France","Canada",1,0,"Group C"),(1986,"Uruguay","Germany",1,1,"Group E"),
    (1986,"Argentina","Italy",1,1,"Group A"),(1986,"France","USSR",2,0,"Group C"),
    (1986,"Brazil","Algeria",1,0,"Group D"),(1986,"Germany","Scotland",2,1,"Group E"),
    (1986,"Argentina","Bulgaria",2,0,"Group A"),(1986,"Mexico","Paraguay",1,1,"Group B"),
    (1986,"Brazil","Poland",4,0,"Quarter-Final"),(1986,"France","Brazil",1,1,"Quarter-Final"),
    (1986,"Germany","Mexico",0,0,"Quarter-Final"),(1986,"Argentina","England",2,1,"Quarter-Final"),
    (1986,"France","Germany",0,2,"Semi-Final"),(1986,"Argentina","Belgium",2,0,"Semi-Final"),
    (1986,"France","Belgium",4,2,"Third place"),(1986,"Argentina","Germany",3,2,"Final"),
    # 1990
    (1990,"Argentina","Cameroon",0,1,"Group B"),(1990,"Italy","Austria",1,0,"Group A"),
    (1990,"Brazil","Sweden",2,1,"Group C"),(1990,"Germany","Yugoslavia",4,1,"Group D"),
    (1990,"Spain","Uruguay",0,0,"Group E"),(1990,"England","Ireland",1,1,"Group F"),
    (1990,"Argentina","USSR",2,0,"Group B"),(1990,"Italy","USA",1,0,"Group A"),
    (1990,"Brazil","Costa Rica",1,0,"Group C"),(1990,"Germany","United Arab Emirates",5,1,"Group D"),
    (1990,"Argentina","Brazil",1,0,"Quarter-Final"),(1990,"Italy","Ireland",1,0,"Quarter-Final"),
    (1990,"Germany","Czechoslovakia",1,0,"Quarter-Final"),(1990,"England","Cameroon",3,2,"Quarter-Final"),
    (1990,"Argentina","Italy",1,1,"Semi-Final"),(1990,"Germany","England",1,1,"Semi-Final"),
    (1990,"Italy","England",2,1,"Third place"),(1990,"Germany","Argentina",1,0,"Final"),
    # 1994
    (1994,"Germany","Bolivia",1,0,"Group C"),(1994,"Brazil","Russia",2,0,"Group B"),
    (1994,"Argentina","Greece",4,0,"Group D"),(1994,"Italy","Ireland",1,0,"Group E"),
    (1994,"Mexico","Norway",1,0,"Group E"),(1994,"USA","Colombia",2,1,"Group A"),
    (1994,"Germany","South Korea",3,2,"Group C"),(1994,"Brazil","Cameroon",3,0,"Group B"),
    (1994,"Netherlands","Ireland",2,0,"Quarter-Final"),(1994,"Brazil","USA",1,0,"Quarter-Final"),
    (1994,"Bulgaria","Germany",2,1,"Quarter-Final"),(1994,"Italy","Spain",2,1,"Quarter-Final"),
    (1994,"Brazil","Sweden",1,0,"Semi-Final"),(1994,"Italy","Bulgaria",2,1,"Semi-Final"),
    (1994,"Sweden","Bulgaria",4,0,"Third place"),(1994,"Brazil","Italy",0,0,"Final"),
    # 1998
    (1998,"Brazil","Scotland",2,1,"Group A"),(1998,"Germany","USA",2,0,"Group F"),
    (1998,"France","South Africa",3,0,"Group C"),(1998,"Argentina","Japan",1,0,"Group H"),
    (1998,"Italy","Chile",2,2,"Group B"),(1998,"Netherlands","Belgium",0,0,"Group E"),
    (1998,"Spain","Nigeria",2,3,"Group D"),(1998,"England","Tunisia",2,0,"Group G"),
    (1998,"France","Saudi Arabia",4,0,"Group C"),(1998,"Brazil","Morocco",3,0,"Group A"),
    (1998,"Germany","Iran",2,0,"Group F"),(1998,"Argentina","England",2,2,"Quarter-Final"),
    (1998,"Netherlands","Argentina",2,1,"Semi-Final"),(1998,"France","Croatia",2,1,"Semi-Final"),
    (1998,"Croatia","Netherlands",2,1,"Third place"),(1998,"France","Brazil",3,0,"Final"),
    # 2002
    (2002,"Senegal","France",1,0,"Group A"),(2002,"Germany","Saudi Arabia",8,0,"Group E"),
    (2002,"South Korea","Poland",2,0,"Group D"),(2002,"Japan","Belgium",2,2,"Group H"),
    (2002,"Argentina","Nigeria",1,0,"Group F"),(2002,"Brazil","Turkey",2,1,"Group C"),
    (2002,"England","Sweden",1,1,"Group F"),(2002,"USA","Portugal",3,2,"Group D"),
    (2002,"Brazil","England",2,1,"Quarter-Final"),(2002,"Germany","USA",1,0,"Quarter-Final"),
    (2002,"South Korea","Spain",0,0,"Quarter-Final"),(2002,"Turkey","Senegal",1,0,"Quarter-Final"),
    (2002,"Germany","South Korea",1,0,"Semi-Final"),(2002,"Brazil","Turkey",1,0,"Semi-Final"),
    (2002,"Turkey","South Korea",3,2,"Third place"),(2002,"Brazil","Germany",2,0,"Final"),
    # 2006
    (2006,"Germany","Costa Rica",4,2,"Group A"),(2006,"England","Paraguay",1,0,"Group B"),
    (2006,"Argentina","Ivory Coast",2,1,"Group C"),(2006,"Mexico","Iran",3,1,"Group D"),
    (2006,"Italy","Ghana",2,0,"Group E"),(2006,"France","Switzerland",0,0,"Group G"),
    (2006,"Brazil","Croatia",1,0,"Group F"),(2006,"Spain","Ukraine",4,0,"Group H"),
    (2006,"Germany","Sweden",2,0,"Quarter-Final"),(2006,"Argentina","Germany",1,1,"Quarter-Final"),
    (2006,"Italy","Ukraine",3,0,"Quarter-Final"),(2006,"France","Brazil",1,0,"Quarter-Final"),
    (2006,"Germany","Italy",0,2,"Semi-Final"),(2006,"France","Portugal",1,0,"Semi-Final"),
    (2006,"Germany","Portugal",3,1,"Third place"),(2006,"Italy","France",1,1,"Final"),
    # 2010
    (2010,"South Africa","Mexico",1,1,"Group A"),(2010,"Germany","Australia",4,0,"Group D"),
    (2010,"Netherlands","Denmark",2,0,"Group E"),(2010,"England","USA",1,1,"Group C"),
    (2010,"Brazil","North Korea",2,1,"Group G"),(2010,"Spain","Switzerland",0,1,"Group H"),
    (2010,"Argentina","Nigeria",1,0,"Group B"),(2010,"France","Uruguay",0,0,"Group A"),
    (2010,"Germany","England",4,1,"Quarter-Final"),(2010,"Argentina","Germany",0,4,"Semi-Final"),
    (2010,"Netherlands","Uruguay",3,2,"Semi-Final"),(2010,"Spain","Germany",1,0,"Semi-Final"),
    (2010,"Germany","Uruguay",3,2,"Third place"),(2010,"Spain","Netherlands",1,0,"Final"),
    # 2014
    (2014,"Brazil","Croatia",3,1,"Group A"),(2014,"Germany","Portugal",4,0,"Group G"),
    (2014,"Netherlands","Spain",5,1,"Group B"),(2014,"Argentina","Bosnia",2,1,"Group F"),
    (2014,"France","Honduras",3,0,"Group E"),(2014,"Colombia","Greece",3,0,"Group C"),
    (2014,"Italy","England",2,1,"Group D"),(2014,"Uruguay","Costa Rica",1,3,"Group D"),
    (2014,"Brazil","Germany",1,7,"Semi-Final"),(2014,"Netherlands","Argentina",0,0,"Semi-Final"),
    (2014,"Brazil","Netherlands",0,3,"Third place"),(2014,"Germany","Argentina",1,0,"Final"),
    (2014,"France","Germany",0,1,"Quarter-Final"),(2014,"Brazil","Colombia",2,1,"Quarter-Final"),
    (2014,"Argentina","Belgium",1,0,"Quarter-Final"),(2014,"Netherlands","Costa Rica",0,0,"Quarter-Final"),
]

TEAM_UNIFICATION = {
    "Germany FR": "Germany",
    "Czechoslovakia": "Czech Republic",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Dutch East Indies": "Indonesia",
    "East Germany": "Germany",
}

def build_dataframes():
    """Construit tous les DataFrames nécessaires."""
    # WorldCups
    worldcups = pd.DataFrame(WORLDCUPS_RAW,
        columns=["Year","Host","QualifiedTeams","Matches","Attendance","Capacity"])

    # Matches
    matches = pd.DataFrame(MATCHES_RAW,
        columns=["Year","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Round"])
    matches["HomeTeam"] = matches["HomeTeam"].replace(TEAM_UNIFICATION)
    matches["AwayTeam"] = matches["AwayTeam"].replace(TEAM_UNIFICATION)
    matches["TotalGoals"] = matches["HomeGoals"] + matches["AwayGoals"]
    matches["Margin"] = abs(matches["HomeGoals"] - matches["AwayGoals"])
    matches["Result"] = matches.apply(
        lambda r: "HomeWin" if r["HomeGoals"] > r["AwayGoals"]
        else ("AwayWin" if r["HomeGoals"] < r["AwayGoals"] else "Draw"), axis=1)
    matches["Phase"] = matches["Round"].apply(
        lambda r: "Groupes" if any(k in str(r).lower() for k in
            ["group","first round","preliminary"]) else "Élimination")

    # Tableau (vue équipe)
    rows = []
    for _, m in matches.iterrows():
        rows.append({
            "Year": m["Year"], "Team": m["HomeTeam"], "Opponent": m["AwayTeam"],
            "Team G": m["HomeGoals"], "Opponent G": m["AwayGoals"], "Phase": m["Phase"],
            "Round": m["Round"]
        })
        rows.append({
            "Year": m["Year"], "Team": m["AwayTeam"], "Opponent": m["HomeTeam"],
            "Team G": m["AwayGoals"], "Opponent G": m["HomeGoals"], "Phase": m["Phase"],
            "Round": m["Round"]
        })
    tableau = pd.DataFrame(rows)
    tableau["Result"] = tableau.apply(
        lambda r: "Win" if r["Team G"] > r["Opponent G"]
        else ("Loss" if r["Team G"] < r["Opponent G"] else "Draw"), axis=1)

    return matches, tableau, worldcups


def build_team_stats(tableau):
    team_stats = tableau.groupby("Team").agg(
        Matchs=("Result","count"),
        Victoires=("Result", lambda x: (x=="Win").sum()),
        Nuls=("Result", lambda x: (x=="Draw").sum()),
        Defaites=("Result", lambda x: (x=="Loss").sum()),
        Buts_marques=("Team G","sum"),
        Buts_encais=("Opponent G","sum"),
        Editions=("Year","nunique"),
    ).reset_index()
    team_stats["Win_rate"] = (team_stats["Victoires"] / team_stats["Matchs"] * 100).round(1)
    team_stats["Buts_pm"] = (team_stats["Buts_marques"] / team_stats["Matchs"]).round(2)
    team_stats["Buts_encais_pm"] = (team_stats["Buts_encais"] / team_stats["Matchs"]).round(2)
    team_stats["Diff_buts"] = team_stats["Buts_marques"] - team_stats["Buts_encais"]
    return team_stats


def build_clusters(team_stats):
    features_cl = ["Win_rate","Buts_pm","Buts_encais_pm","Diff_buts","Matchs"]
    X_cl = team_stats[features_cl].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cl)

    inertias = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    K_OPTIMAL = 4
    kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
    team_stats = team_stats.copy()
    team_stats["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_means = team_stats.groupby("Cluster")[features_cl].mean()
    cluster_names = {}
    for c in range(K_OPTIMAL):
        wr = cluster_means.loc[c, "Win_rate"]
        if wr >= 55:   cluster_names[c] = "🏆 Dominants"
        elif wr >= 42: cluster_names[c] = "💪 Solides"
        elif wr >= 28: cluster_names[c] = "⚡ Challengers"
        else:          cluster_names[c] = "🌱 Émergents"

    team_stats["Profil"] = team_stats["Cluster"].map(cluster_names)
    return team_stats, list(range(2, 10)), inertias


def build_ml_model(team_stats, tableau):
    df_model = tableau.merge(
        team_stats[["Team","Win_rate","Buts_pm","Buts_encais_pm","Diff_buts"]], on="Team"
    ).merge(
        team_stats[["Team","Win_rate","Buts_pm","Buts_encais_pm","Diff_buts"]].rename(columns={
            "Team":"Opponent","Win_rate":"Opp_wr","Buts_pm":"Opp_bpm",
            "Buts_encais_pm":"Opp_bencais","Diff_buts":"Opp_diff"
        }), on="Opponent"
    )
    df_model["label"] = df_model["Result"].map({"Win":2,"Draw":1,"Loss":0})

    features_m = ["Win_rate","Buts_pm","Buts_encais_pm","Diff_buts",
                  "Opp_wr","Opp_bpm","Opp_bencais","Opp_diff"]
    X_m = df_model[features_m].dropna()
    y_m = df_model.loc[X_m.index, "label"]

    X_train, X_test, y_train, y_test = train_test_split(X_m, y_m, test_size=0.2, random_state=42)
    sc_m = StandardScaler()
    X_train_s = sc_m.fit_transform(X_train)
    X_test_s = sc_m.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)
    hgb.fit(X_train_s, y_train)

    return clf, hgb, sc_m, X_test_s, y_test, features_m


def predict_match(team_a, team_b, team_stats, clf, sc_m):
    if team_a not in team_stats["Team"].values:
        return None
    if team_b not in team_stats["Team"].values:
        return None
    ra = team_stats[team_stats["Team"]==team_a].iloc[0]
    rb = team_stats[team_stats["Team"]==team_b].iloc[0]
    feat_vec = [[ra["Win_rate"], ra["Buts_pm"], ra["Buts_encais_pm"], ra["Diff_buts"],
                 rb["Win_rate"], rb["Buts_pm"], rb["Buts_encais_pm"], rb["Diff_buts"]]]
    feat_s = sc_m.transform(feat_vec)
    proba = clf.predict_proba(feat_s)[0]
    return {"team_a": team_a, "team_b": team_b,
            "p_win_a": proba[2], "p_draw": proba[1], "p_win_b": proba[0]}
