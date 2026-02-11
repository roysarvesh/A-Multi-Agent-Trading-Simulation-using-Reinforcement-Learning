import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
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
# GLOBAL CSS (PREMIUM THEME)
# ==================================================
st.markdown("""
<style>
html, body {
    background: #0a0f14 !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f14 0%, #111821 80%);
    color: #cccccc;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 { color: #EAEAEA !important; font-weight: 700; }
.metric-card {
    background: rgba(255, 255, 255, 0.06);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0px 0px 12px rgba(0,0,0,0.5);
}
.metric-title { font-size: 11px; opacity: 0.7; text-transform: uppercase; }
.metric-value { font-size: 26px; font-weight: 700; color: white; }
.alert-box {
    background: rgba(255, 255, 255, 0.10);
    padding: 12px;
    border-left: 4px solid #ff4b4b;
    border-radius: 6px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙️ Dashboard Controls")

asset = st.sidebar.selectbox(
    "Select Asset",
    ["AAPL", "TSLA", "BTC-USD"]
)

selected_agent = st.sidebar.selectbox(
    "Select Agent",
    ["conservative", "aggressive", "momentum", "mean_reversion"]
)

show_battle = st.sidebar.checkbox("Multi-Agent Battle Mode")
show_animation = st.sidebar.checkbox("Auto Replay Animation")
show_slider_replay = st.sidebar.checkbox("Drag-to-Replay Slider")
show_heatmap = st.sidebar.checkbox("Show Agent Correlation Heatmap")
show_risk = st.sidebar.checkbox("Show Risk Heatmaps")
show_forecast = st.sidebar.checkbox("Enable Price Forecasting (LSTM)")
chart_style = st.sidebar.selectbox("Chart Theme", ["Dark", "Neon", "Cyber", "Classic"])

# ==================================================
# LOAD DATA
# ==================================================
df = get_market_data(asset, "2020-01-01", "2024-01-01")

# ==================================================
# LOAD MODELS
# ==================================================
AGENTS = ["conservative", "aggressive", "momentum", "mean_reversion"]
MODELS = {a: PPO.load(f"{a}_ppo_model") for a in AGENTS}

# ==================================================
# EVALUATE AGENT
# ==================================================
model = MODELS[selected_agent]
portfolio_values, metrics, buy_points, sell_points = evaluate_agent_detailed(
    model, selected_agent, df
)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    f"<h1 style='text-align:center;'>🤖💹 Multi-Agent RL Trading Dashboard — {asset}</h1>",
    unsafe_allow_html=True
)

# ==================================================
# METRIC CARDS
# ==================================================
st.markdown("### 📊 Performance Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

cards = [col1, col2, col3, col4, col5]
titles = ["Final Portfolio", "Sharpe Ratio", "Max Drawdown", "Volatility", "Win Rate"]
values = [
    metrics["Final Value"],
    metrics["Sharpe Ratio"],
    metrics["Max Drawdown"],
    metrics["Volatility"],
    metrics["Win Rate"]
]

for c, t, v in zip(cards, titles, values):
    c.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>{t}</div>
        <div class='metric-value'>{v:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# PRICE CHART (CANDLESTICK + BUY/SELL)
# ==================================================
st.markdown("### 🕯️ Candlestick Chart with Trade Signals")

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
        marker=dict(color="lime", size=12, symbol="triangle-up"),
        name="BUY"
    ))

if sell_points:
    x_s, y_s = zip(*sell_points)
    fig.add_trace(go.Scatter(
        x=df.index[list(x_s)],
        y=y_s,
        mode="markers",
        marker=dict(color="red", size=12, symbol="triangle-down"),
        name="SELL"
    ))

template = "plotly_dark" if chart_style != "Classic" else "plotly_white"
fig.update_layout(template=template, height=600)
st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TRADING JOURNAL EXPORT
# ==================================================
st.markdown("### 📘 Trading Journal Export")

trade_df = pd.DataFrame({
    "Step": [s for s, _ in buy_points + sell_points],
    "Price": [p for _, p in buy_points + sell_points],
    "Type": ["BUY"] * len(buy_points) + ["SELL"] * len(sell_points)
})

col_csv, col_xlsx = st.columns(2)

col_csv.download_button(
    "📥 Download CSV",
    trade_df.to_csv().encode("utf-8"),
    file_name=f"{selected_agent}_journal.csv"
)

col_xlsx.download_button(
    "📥 Download Excel",
    trade_df.to_excel("journal.xlsx"),
    file_name=f"{selected_agent}_journal.xlsx"
)

# ==================================================
# RISK HEATMAPS (VaR / CVaR)
# ==================================================
if show_risk:
    st.markdown("### ⚠️ Risk Heatmaps (VaR & CVaR)")

    returns = np.diff(portfolio_values)
    VaR_95 = np.percentile(returns, 5)
    CVaR_95 = returns[returns < VaR_95].mean()

    st.warning(f"📌 **VaR(95%)**: {VaR_95:.4f}")
    st.error(f"📌 **CVaR(95%)**: {CVaR_95:.4f}")

    fig_risk, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(np.corrcoef([returns, np.abs(returns)]), annot=True, cmap="inferno", ax=ax)
    st.pyplot(fig_risk)

# ==================================================
# AGENT CORRELATION HEATMAP
# ==================================================
if show_heatmap:
    st.markdown("### 🔥 Agent Correlation Heatmap")

    battle = multi_agent_battle(df)
    min_len = min(len(v) for v in battle.values())
    df_corr = pd.DataFrame({a: battle[a][:min_len] for a in battle})

    fig_h, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_h)

# ==================================================
# MULTI-AGENT BATTLE MODE
# ==================================================
if show_battle:
    st.markdown("### ⚔️ Multi-Agent Battle Arena")

    battle_hist = multi_agent_battle(df)

    fig_b = go.Figure()
    for agent, curve in battle_hist.items():
        fig_b.add_trace(go.Scatter(
            x=list(range(len(curve))),
            y=curve,
            mode="lines",
            name=agent
        ))

    fig_b.update_layout(template=template, height=450)
    st.plotly_chart(fig_b, use_container_width=True)

    winner = max({a: v[-1] for a, v in battle_hist.items()}, key=lambda x: battle_hist[x][-1])
    st.success(f"🏆 Winner: **{winner}**")

# ==================================================
# DRAG-TO-REPLAY SLIDER
# ==================================================
if show_slider_replay:
    st.markdown("### 🎚️ Interactive Replay Slider")

    step = st.slider("Replay Position", 30, len(portfolio_values)-1, 100)

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=np.arange(step),
        y=df["Close"][:step],
        name="Market",
        line=dict(color="#4e89ff")
    ))
    fig_r.add_trace(go.Scatter(
        x=np.arange(step),
        y=portfolio_values[:step],
        name="Portfolio",
        line=dict(color="#00ffa2")
    ))

    fig_r.update_layout(template=template, height=420)
    st.plotly_chart(fig_r, use_container_width=True)

# ==================================================
# REAL-TIME PLAYBACK
# ==================================================
if show_animation:
    st.markdown("### 🎥 Live Replay Animation")

    placeholder = st.empty()

    for i in range(20, len(portfolio_values)):
        with placeholder.container():

            fig_rt = go.Figure()
            fig_rt.add_trace(go.Scatter(
                x=np.arange(i),
                y=df["Close"][:i],
                name="Market",
                line=dict(color="#4e89ff")
            ))
            fig_rt.add_trace(go.Scatter(
                x=np.arange(i),
                y=portfolio_values[:i],
                name="Portfolio",
                line=dict(color="#00ffa2")
            ))

            fig_rt.update_layout(template=template, height=420)
            st.plotly_chart(fig_rt, use_container_width=True)

        time.sleep(0.02)

    st.success("🎬 Replay Completed!")

# ==================================================
# LSTM PRICE FORECASTING
# ==================================================
if show_forecast:
    st.markdown("### 📈 LSTM Price Forecast (Next 30 Days)")

    prices = df["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    window = 60

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model_lstm = LSTMNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)

    for epoch in range(10):  
        optimizer.zero_grad()
        pred = model_lstm(X_torch)
        loss = criterion(pred, y_torch)
        loss.backward()
        optimizer.step()

    last_seq = scaled[-window:]
    preds = []
    seq = last_seq.copy()

    for _ in range(30):
        inp = torch.tensor(seq.reshape(1, window, 1), dtype=torch.float32)
        pred = model_lstm(inp).detach().numpy()
        preds.append(pred)
        seq = np.vstack([seq[1:], pred])

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Historical Price"
    ))
    future_index = pd.date_range(df.index[-1], periods=31, freq="D")[1:]
    fig_fc.add_trace(go.Scatter(
        x=future_index,
        y=forecast.flatten(),
        name="Forecast (30 Days)",
        line=dict(color="orange")
    ))
    fig_fc.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_fc, use_container_width=True)

    st.info("📢 Forecast generated using a basic LSTM model. You can upgrade it to multi-feature forecasting or deeper LSTM layers.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;opacity:0.6;'>Made with ❤️ by Sarvesh — AI Trading Agents Powered by RL, PPO, LSTM & Streamlit</p>",
    unsafe_allow_html=True
)
