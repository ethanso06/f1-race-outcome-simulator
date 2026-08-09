# python -m streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np

from XGB_model import load_mu_model, predict_mu
from RF_model import load_sigma_model, predict_sigma
from monte_carlo import simulate_race_outcome, result_df
from feature_engineering import load_features

ACCENT_RED = "#d3131b"
CARD = "#111a2b"
TEXT = "#e8eef5"
MUTED = "#9fb3c8"
ROW_ALT = "#0d1526"


@st.cache_resource
def get_models():
    mu_pipeline = load_mu_model()
    sigma_pipeline = load_sigma_model()
    residuals = np.load('models/mu_residuals.npy')
    return mu_pipeline, sigma_pipeline, residuals


@st.cache_data
def get_all_features():
    return load_features([2022, 2023, 2024, 2025, 2026])

import fastf1

@st.cache_data
def get_event_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule[['RoundNumber', 'EventName', 'Location', 'Country']]


def set_styles():
    st.set_page_config(page_title="F1 Race Outcome Simulator", layout="wide")
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(145deg, #0b1220 0%, #0f192d 60%, #0b1220 100%);
            color: {TEXT};
        }}
        [data-testid="stSidebar"] {{ background: #0a1020; }}
        h1, h2, h3 {{ color: {TEXT}; font-weight: 800; }}
        .subtitle {{ color: {MUTED}; font-size: 0.95rem; margin-top: -0.3rem; }}
        .card {{
            background: {CARD}; padding: 1rem 1.2rem; border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.06); color: {MUTED};
        }}
        .leaderboard {{
            width: 100%;
            border-collapse: collapse;
            background: {CARD};
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .leaderboard th {{
            background: {ACCENT_RED};
            color: white;
            text-align: left;
            padding: 10px 14px;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
        }}
        .leaderboard td {{
            padding: 9px 14px;
            font-size: 0.9rem;
            color: {TEXT};
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .leaderboard tr:nth-child(even) {{ background: {ROW_ALT}; }}
        .pos-badge {{
            font-weight: 800;
            color: {ACCENT_RED};
        }}
        .driver-name {{ font-weight: 700; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_sidebar(all_features: pd.DataFrame):
    st.sidebar.markdown("### Race Setup")

    years_available = sorted(
        all_features.loc[all_features['year'] == 2026, 'year'].unique(),
        reverse=True
    )
    year = st.sidebar.selectbox("Year", years_available, index=0)

    rounds_available = sorted(
        all_features.loc[all_features['year'] == year, 'round_num'].unique()
    )
    schedule = get_event_schedule(year)
    round_to_name = dict(zip(schedule['RoundNumber'], schedule['EventName']))

    round_num = st.sidebar.selectbox(
        "Round", rounds_available,
        format_func=lambda r: f"R{r} — {round_to_name.get(r, 'Unknown')}",
        index=0,
    )

    event_name = round_to_name.get(round_num, f"Round {round_num}")

    n_sims = st.sidebar.slider("Monte Carlo simulations", 500, 50000, 10000, step=500)
    run = st.sidebar.button("Run Simulation", use_container_width=True)

    return {
        "year": year,
        "round_num": round_num,
        "event_name": event_name,
        "n_sims": n_sims,
        "run": run,
    }


def get_race_features(all_features: pd.DataFrame, year: int, round_num: int) -> pd.DataFrame:
    mask = (all_features['year'] == year) & (all_features['round_num'] == round_num)
    return all_features[mask].copy()


def run_simulation(race_df: pd.DataFrame, n_sims: int):
    mu_pipeline, sigma_pipeline, residuals = get_models()

    driver_names = race_df['driver'].tolist()
    X = race_df.drop(columns=['finish_position', 'race_points'], errors='ignore')

    mu = predict_mu(X, mu_pipeline)
    sigma = predict_sigma(X, sigma_pipeline, residual_fallback=np.std(residuals))

    pred_positions = simulate_race_outcome(mu, sigma, residuals, n_sims=n_sims)
    results = result_df(pred_positions, driver_names)
    results = results.sort_values('expected position').reset_index(drop=True)
    return results


def render_leaderboard(results: pd.DataFrame) -> str:
    rows_html = []
    for i, row in results.iterrows():
        pos = i + 1
        rows_html.append(
            f"<tr>"
            f"<td class='pos-badge'>P{pos}</td>"
            f"<td class='driver-name'>{row['Drivers']}</td>"
            f"<td>{row['expected position']:.2f}</td>"
            f"<td>{row['win probability'] * 100:.1f}%</td>"
            f"<td>{row['podium probability'] * 100:.1f}%</td>"
            f"<td>{row['points probability'] * 100:.1f}%</td>"
            f"</tr>"
        )

    table_html = (
        "<table class='leaderboard'>"
        "<thead><tr>"
        "<th>POS</th><th>DRIVER</th><th>EXPECTED POS</th>"
        "<th>WIN %</th><th>PODIUM %</th><th>POINTS %</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )
    return table_html


def main():
    set_styles()
    all_features = get_all_features()
    settings = build_sidebar(all_features)

    st.markdown(
        """
        <div class="card">
            <h1 style="margin-bottom:0;"> F1 Race Outcome Simulator</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.session_state["results"] = None

    if settings["run"]:
        race_df = get_race_features(all_features, settings["year"], settings["round_num"])
        if race_df.empty:
            st.error("No data found for this year/round combination.")
        else:
            with st.spinner("Running Monte Carlo simulation..."):
                st.session_state["results"] = run_simulation(race_df, settings["n_sims"])

    results = st.session_state["results"]

    if results is not None:
        st.markdown(f"#### {settings['event_name']} {settings['year']} — Prediction")
        st.markdown(render_leaderboard(results), unsafe_allow_html=True)
    else:
        st.markdown("#### Leaderboard")
        st.markdown('<div class="card">Run a simulation to see the leaderboard.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()