import fastf1
from fastf1 import Cache
from fastf1.ergast import Ergast
import pandas as pd
import numpy as np
import os

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f1_data')
os.makedirs(cache_dir, exist_ok=True)
Cache.enable_cache(cache_dir)
print("CACHE PATH IN USE:", cache_dir)

ergast = Ergast()

def compute_volatility(feature_df: pd.DataFrame, window = 5):
    df = feature_df.copy()
    df = df.sort_values(['driver', 'year', 'round_num'])
    df['volatility'] = (
        df.groupby('driver')['finish_position']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
    )
    return df

def compute_team_volatility(feature_df: pd.DataFrame, window = 10):
    df = feature_df.copy()
    df = df.sort_values(['team', 'year', 'round_num', 'driver'])
    df['team_volatility'] = (
        df.groupby('team')['finish_position']
        .transform(lambda x : x.shift(1).rolling(window, min_periods = 6 ).std())
    )
    return df

def quali_times(year, round_num):

    try:
        quali_result = ergast.get_qualifying_results(season=year, round=round_num)
        df = quali_result.content[0]
    except Exception as e:
        print(f"Ergast quali fallback unavailable {year} round {round_num}: {e}")
        return {}

    #find best quali times of each driver
    best_quali_times = {}
    for _, row in df.iterrows():
        for col in ('Q3', 'Q2', 'Q1'):
            val = row.get(col)
            if pd.notna(val):
                best_quali_times[row['driverCode']] = val.total_seconds()
                break
    return best_quali_times

def teammate_quali_delta(feature_df: pd.DataFrame):
    df = feature_df.copy()
    df["teammate_gap_quali_time"] = np.nan

    for (year, round_num, team), group in df.groupby(['year', 'round_num', 'team']): 
        if len(group) != 2: #(skip incomplete teams)
            continue 

        idx1, idx2 = group.index
        val1 = df.loc[idx1, 'quali_time'] # df.loc[row, column]
        val2 = df.loc[idx2, 'quali_time']
        df.loc[idx1, 'teammate_gap_quali_time'] = val1 - val2 
        df.loc[idx2, 'teammate_gap_quali_time'] = val2 - val1

    return df



    
def build_features_df(years, sort=True):
    rows = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for round_num in schedule['RoundNumber']:
            try:
                event = fastf1.get_event(year, round_num)

                race = event.get_race()
                race.load(laps = False, telemetry=False, weather=True)

                quali = event.get_qualifying()
                try:
                    quali.load(laps = True, telemetry=False, weather=True)
                    quali_loaded = True
                except Exception as e:
                    print(f"Qualifying telemetry unavailable {year} round {round_num}, falling back to Ergast: {e}")
                    quali_loaded = False

                sprint_df = None
                if event['EventFormat'] != 'conventional':
                    try:
                        sprint = event.get_sprint()
                        sprint.load(laps = False, telemetry=False, weather=True)
                        sprint_df = sprint.results[['Abbreviation', 'Position', 'Points']].rename(
                            columns={'Position': 'sprint_position', 'Points': 'sprint_points'}
                        )
                    except Exception as e:
                        print(f"Sprint unavailable {year} round {round_num}: {e}")
                
                try:
                    if quali_loaded and quali.weather_data is not None and not quali.weather_data.empty:
                        rainfall = quali.weather_data['Rainfall'].any()
                        track_temp = quali.weather_data["TrackTemp"].mean()
                    else:
                        # unknown (not "confirmed dry"), not the same as rainfall=False
                        rainfall = np.nan
                        track_temp = np.nan # XGB handles the nan by imputation
                except Exception:
                    rainfall = np.nan
                    track_temp = np.nan


                

                laps = pd.DataFrame()
                compound_share = pd.DataFrame()
                track_chaos = pd.Series(dtype=float)
                if quali_loaded:
                    try:
                        laps = quali.laps
                        # fetch data from quali, not race
                        compound_share = laps.groupby(["Driver", "Compound"]).size().unstack(fill_value=0)
                        compound_share = compound_share.div(compound_share.sum(axis=1), axis=0) # express in fraction

                        track_chaos = (
                            laps.groupby("Driver")["TrackStatus"]
                            .apply(lambda x: x.fillna("1").astype(str).str.contains("[2-9]").sum())
                        )
                    except Exception:
                        laps = pd.DataFrame()
                        compound_share = pd.DataFrame()
                        track_chaos = pd.Series(dtype=float)

                # only hits the network if the livetiming laps above weren't available
                if not laps.empty:
                    ergast_quali_times = {}
                else:
                    ergast_quali_times = quali_times(year, round_num)

                results = race.results
                for _, r in results.iterrows():
                    driver = r['Abbreviation']

                    if not laps.empty:
                        quali_laps = laps.pick_drivers(driver)
                        quali_fastest = quali_laps.pick_fastest()
                        if quali_fastest is not None and pd.notna(quali_fastest['LapTime']):
                            quali_time = quali_fastest['LapTime'].total_seconds()
                        else:
                            quali_time = np.nan
                    else:
                        quali_time = ergast_quali_times.get(driver, np.nan)

                    sprint_position = np.nan
                    sprint_points = 0.0
                    if sprint_df is not None:
                        match = sprint_df[sprint_df['Abbreviation'] == driver]
                        if not match.empty:
                            sprint_position = match['sprint_position'].values[0]
                            sprint_points = match['sprint_points'].values[0]

                    driver_compounds = (
                        compound_share.loc[driver].to_dict() if driver in compound_share.index else {}
                    )
                    driver_track_chaos = (
                        track_chaos.loc[driver] if driver in track_chaos.index else np.nan
                    )

                    rows.append({
                        'year': year,
                        'round_num': round_num,
                        'driver': driver,
                        'team': r['TeamName'],
                        'grid_position': r['GridPosition'],
                        'quali_time': quali_time,
                        'is_sprint_weekend': sprint_df is not None,
                        'sprint_position': sprint_position,
                        'sprint_points': sprint_points,
                        'rainfall' : rainfall,
                        'track_temp' : track_temp,
                        'track_chaos' : driver_track_chaos,
                        **{f'compound_frac{k}': v for k, v in driver_compounds.items()},
                        # ** is an unpacking operator
                        'finish_position': r['Position'],
                        'race_points': r['Points'],
                    })
            except Exception as e:
                import traceback
                print(f"Skipped {year} round {round_num}: {e}")
                traceback.print_exc()

    df = pd.DataFrame(rows)
    if sort and not df.empty:
        df = df.sort_values(['year', 'round_num']).reset_index(drop=True)
    return df


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def load_features(years, path=None, force_rebuild=False):
    if path is None:
        os.makedirs(data_dir, exist_ok=True)
        years_tag = "-".join(str(y) for y in years)
        path = os.path.join(data_dir, f"features_{years_tag}.csv")

    if not force_rebuild and os.path.exists(path):
        return pd.read_csv(path)

    df = build_features_df(years)
    df = compute_volatility(df)
    df = compute_team_volatility(df)
    df = teammate_quali_delta(df)
    df = df.sort_values(["year", "round_num", "finish_position"]).reset_index(drop = True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    load_features([2022, 2023, 2024, 2025, 2026])
  






