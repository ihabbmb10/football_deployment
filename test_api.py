import requests

url = "http://127.0.0.1:5000/predict"

data = {
  "rank_change_home": 3.0,
  "rank_change_away": -2.0,
  "home_goals_mean": 1.8,
  "home_goals_mean_l5": 2.1,
  "home_goals_suf_mean": 0.9,
  "home_goals_suf_mean_l5": 1.0,
  "home_rank_mean": 15.0,
  "home_rank_mean_l5": 14.0,
  "home_points_mean": 1500.0,
  "home_points_mean_l5": 1510.0,
  "away_goals_mean": 1.2,
  "away_goals_mean_l5": 1.0,
  "away_goals_suf_mean": 1.4,
  "away_goals_suf_mean_l5": 1.5,
  "away_rank_mean": 55.0,
  "away_rank_mean_l5": 57.0,
  "away_points_mean": 1200.0,
  "away_points_mean_l5": 1180.0
}

response = requests.post(url, json=data)

print(response.json())