import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO

from your_functions import (
    get_market_data,
    evaluate_agent_detailed,
    multi_agent_battle
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="AI Multi-Agent Trading Dashboard",
    layout="wide",
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙ Dashboard Controls")

asset = st.sidebar.selectbox(
    "Select Asset",
    ["AAPL", "TSLA", "BTC-USD"]
)

selected_agent = st.sidebar.selectbox(
    "Select Agent",
    ["conservative", "aggressive", "momentum", "mean_reversion"]
)

show_battle = st.sidebar.checkbox("Multi-Agent Battle")
show_heatmap = st.sidebar.checkbox("Agent Correlation Heatmap")
show_risk = st.sidebar.checkbox("Risk Metrics")
show_forecast = st.sidebar.checkbox("LSTM Forecast")

# ==================================================
# LOAD DATA (SAFE)
# ==================================================
@st.cache_data
def load_data(asset):
    df = get_market_data(asset, "2020-01-01", "2024-01-01")
    return df

df = load_data(asset)

if df is None or df.empty:
    st.error("❌ Failed to load market data. Try switching asset.")
    st.stop()

# ==================================================
# LOAD MODELS SAFELY
# ==================================================
AGENTS = ["conservative", "aggressive", "momentum", "mean_reversion"]
MODELS = {}

for a in AGENTS:
    model_path = f"models/{a}_ppo_model.zip"
    if os.path.exists(model_path):
        MODELS[a] = PPO.load(model_path)

if selected_agent not in MODELS:
    st.error("Model not found.")
    st.stop()

model = MODELS[selected_agent]

# ==================================================
# HEADER
# ==================================================
st.title(f"🤖 Multi-Agent RL Trading Dashboard — {asset}")

# ==================================================
# MANUAL TRADING CONTROLS (NEW FEATURE)
# ==================================================
st.subheader("💰 Manual Asset Trading")

if "manual_balance" not in st.session_state:
    st.session_state.manual_balance = 10000
    st.session_state.manual_holdings = 0

current_price = df.iloc[-1]["Close"]

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Balance", f"${st.session_state.manual_balance:.2f}")
col3.metric("Holdings", st.session_state.manual_holdings)

buy_col, sell_col = st.columns(2)

if buy_col.button("🟢 Buy 1 Unit"):
    if st.session_state.manual_balance >= current_price:
        st.session_state.manual_balance -= current_price
        st.session_state.manual_holdings += 1

if sell_col.button("🔴 Sell 1 Unit"):
    if st.session_state.manual_holdings > 0:
        st.session_state.manual_balance += current_price
        st.session_state.manual_holdings -= 1

manual_value = (
    st.session_state.manual_balance +
    st.session_state.manual_holdings * current_price
)

st.success(f"📊 Total Portfolio Value: ${manual_value:.2f}")

# ==================================================
# EVALUATE RL AGENT
# ==================================================
portfolio_values, metrics, buy_points, sell_points = evaluate_agent_detailed(
    model, selected_agent, df
)

# ==================================================
# METRICS
# ==================================================
st.subheader("📊 RL Agent Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Final Value", f"${metrics['Final Value']:.2f}")
col2.metric("Sharpe", f"{metrics['Sharpe Ratio']:.3f}")
col3.metric("Max DD", f"{metrics['Max Drawdown']:.3f}")
col4.metric("Volatility", f"{metrics['Volatility']:.3f}")
col5.metric("Win Rate", f"{metrics['Win Rate']:.2%}")

# ==================================================
# PRICE CHART
# ==================================================
st.subheader("📈 Price Chart + Agent Trades")

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

fig.update_layout(template="plotly_dark", height=550)
st.plotly_chart(fig, use_container_width=True)

# ==================================================
# RISK METRICS
# ==================================================
if show_risk:
    st.subheader("⚠ Risk Metrics")

    returns = np.diff(portfolio_values)
    VaR = np.percentile(returns, 5)
    CVaR = returns[returns < VaR].mean()

    st.warning(f"VaR (95%): {VaR:.4f}")
    st.error(f"CVaR (95%): {CVaR:.4f}")

# ==================================================
# MULTI AGENT BATTLE
# ==================================================
if show_battle:
    st.subheader("⚔ Multi-Agent Battle")

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
# HEATMAP
# ==================================================
if show_heatmap:
    st.subheader("🔥 Agent Correlation")

    battle_hist = multi_agent_battle(df, MODELS)

    min_len = min(len(v) for v in battle_hist.values())
    df_corr = pd.DataFrame({a: battle_hist[a][:min_len] for a in battle_hist})

    fig_h, ax = plt.subplots()
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_h)

# ==================================================
# LSTM FORECAST
# ==================================================
if show_forecast:
    st.subheader("📈 LSTM Forecast")

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
st.markdown("Made with ❤️ by Sarvesh — Production Ready AI Trading Platform")
