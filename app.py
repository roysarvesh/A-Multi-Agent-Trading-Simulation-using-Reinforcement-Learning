import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import os
import io

from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO

# ==================================================
# IMPORT CUSTOM FUNCTIONS
# ==================================================
from your_functions import (
    get_market_data,
    SingleAgentWrapper,
    evaluate_agent_detailed,
    multi_agent_battle
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="AI Multi-Agent Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# THEME
# ==================================================
st.markdown("""
<style>
html, body { background: #0a0f14 !important; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f14 0%, #111821 80%);
    color: #cccccc;
}
.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}
.metric-title { font-size: 11px; opacity: 0.7; text-transform: uppercase; }
.metric-value { font-size: 24px; font-weight: 700; color: white; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙️ Dashboard Controls")

asset = st.sidebar.selectbox("Select Asset", ["AAPL", "TSLA", "BTC-USD"])
selected_agent = st.sidebar.selectbox(
    "Select Agent",
    ["conservative", "aggressive", "momentum", "mean_reversion"]
)

show_battle = st.sidebar.checkbox("Multi-Agent Battle Mode")
show_animation = st.sidebar.checkbox("Auto Replay Animation")
show_slider_replay = st.sidebar.checkbox("Replay Slider")
show_heatmap = st.sidebar.checkbox("Agent Correlation Heatmap")
show_risk = st.sidebar.checkbox("Risk Heatmaps")
show_forecast = st.sidebar.checkbox("Enable LSTM Forecast")

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data(asset):
    return get_market_data(asset, "2020-01-01", "2024-01-01")

df = load_data(asset)

# ==================================================
# LOAD MODELS SAFELY
# ==================================================
AGENTS = ["conservative", "aggressive", "momentum", "mean_reversion"]
MODELS = {}

for a in AGENTS:
    model_path = f"models/{a}_ppo_model.zip"
    if os.path.exists(model_path):
        MODELS[a] = PPO.load(model_path)
    else:
        st.warning(f"⚠ Model not found: {model_path}")

if selected_agent not in MODELS:
    st.error("Selected agent model not found.")
    st.stop()

model = MODELS[selected_agent]

# ==================================================
# EVALUATE SELECTED AGENT
# ==================================================
portfolio_values, metrics, buy_points, sell_points = evaluate_agent_detailed(
    model, selected_agent, df
)

# ==================================================
# HEADER
# ==================================================
st.markdown(f"# 🤖 Multi-Agent RL Trading Dashboard — {asset}")

# ==================================================
# METRICS
# ==================================================
st.markdown("### 📊 Performance Metrics")

cols = st.columns(5)
metric_titles = ["Final Value", "Sharpe", "Max DD", "Volatility", "Win Rate"]
metric_vals = [
    metrics["Final Value"],
    metrics["Sharpe Ratio"],
    metrics["Max Drawdown"],
    metrics["Volatility"],
    metrics["Win Rate"]
]

for col, title, val in zip(cols, metric_titles, metric_vals):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value'>{val:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# CANDLESTICK CHART
# ==================================================
st.markdown("### 🕯️ Price Chart with Signals")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index[:len(portfolio_values)],
    open=df["Open"][:len(portfolio_values)],
    high=df["High"][:len(portfolio_values)],
    low=df["Low"][:len(portfolio_values)],
    close=df["Close"][:len(portfolio_values)],
    name="Market"
))

if buy_points:
    x_b, y_b = zip(*buy_points)
    fig.add_trace(go.Scatter(
        x=df.index[list(x_b)],
        y=y_b,
        mode="markers",
        marker=dict(color="lime", size=8),
        name="BUY"
    ))

if sell_points:
    x_s, y_s = zip(*sell_points)
    fig.add_trace(go.Scatter(
        x=df.index[list(x_s)],
        y=y_s,
        mode="markers",
        marker=dict(color="red", size=8),
        name="SELL"
    ))

fig.update_layout(template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TRADING JOURNAL EXPORT (NO XLSXWRITER DEPENDENCY)
# ==================================================
st.markdown("### 📘 Trading Journal")

trades = buy_points + sell_points

if trades:
    trade_df = pd.DataFrame({
        "Step": [t[0] for t in trades],
        "Price": [t[1] for t in trades],
        "Type": ["BUY"] * len(buy_points) + ["SELL"] * len(sell_points)
    })

    csv = trade_df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, f"{selected_agent}_journal.csv")

# ==================================================
# RISK METRICS
# ==================================================
if show_risk:
    st.markdown("### ⚠ Risk Analysis")

    returns = np.diff(portfolio_values)
    VaR = np.percentile(returns, 5)
    CVaR = returns[returns < VaR].mean()

    st.warning(f"VaR(95%): {VaR:.4f}")
    st.error(f"CVaR(95%): {CVaR:.4f}")

# ==================================================
# AGENT CORRELATION HEATMAP
# ==================================================
if show_heatmap and MODELS:
    st.markdown("### 🔥 Agent Correlation")

    battle = multi_agent_battle(df, MODELS)

    min_len = min(len(v) for v in battle.values())
    df_corr = pd.DataFrame({a: battle[a][:min_len] for a in battle})

    fig_h, ax = plt.subplots()
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_h)

# ==================================================
# MULTI-AGENT BATTLE
# ==================================================
if show_battle and MODELS:
    st.markdown("### ⚔ Multi-Agent Battle")

    battle_hist = multi_agent_battle(df, MODELS)

    fig_b = go.Figure()

    for agent, curve in battle_hist.items():
        fig_b.add_trace(go.Scatter(
            x=np.arange(len(curve)),
            y=curve,
            mode="lines",
            name=agent
        ))

    fig_b.update_layout(template="plotly_dark")
    st.plotly_chart(fig_b, use_container_width=True)

# ==================================================
# REPLAY SLIDER
# ==================================================
if show_slider_replay:
    step = st.slider("Replay Step", 30, len(portfolio_values)-1, 100)

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=np.arange(step),
        y=df["Close"][:step],
        name="Market"
    ))
    fig_r.add_trace(go.Scatter(
        x=np.arange(step),
        y=portfolio_values[:step],
        name="Portfolio"
    ))

    fig_r.update_layout(template="plotly_dark")
    st.plotly_chart(fig_r, use_container_width=True)

# ==================================================
# LSTM FORECAST
# ==================================================
if show_forecast:
    st.markdown("### 📈 LSTM Forecast (30 Days)")

    prices = df["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    window = 60

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.fc = nn.Linear(32,1)
        def forward(self,x):
            out,_ = self.lstm(x)
            return self.fc(out[:,-1,:])

    model_lstm = LSTMNet()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    for _ in range(5):
        optimizer.zero_grad()
        pred = model_lstm(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    seq = scaled[-window:]
    preds = []

    for _ in range(30):
        inp = torch.tensor(seq.reshape(1,window,1), dtype=torch.float32)
        pred = model_lstm(inp).detach().numpy()
        preds.append(pred)
        seq = np.vstack([seq[1:], pred])

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    future_dates = pd.date_range(df.index[-1], periods=31, freq="D")[1:]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical"))
    fig_fc.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), name="Forecast"))
    fig_fc.update_layout(template="plotly_dark")
    st.plotly_chart(fig_fc, use_container_width=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown("Made with ❤️ by Sarvesh — RL + PPO + LSTM + Streamlit")
