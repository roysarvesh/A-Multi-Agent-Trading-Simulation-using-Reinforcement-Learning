
# 📘 **AI Multi-Agent Reinforcement Learning Trading System**

```markdown
# 🤖📈 AI Multi-Agent Reinforcement Learning Trading System  
*A complete multi-agent trading ecosystem with PPO, LSTM forecasting, PettingZoo, and a full interactive Streamlit dashboard.*

---

## 🚀 Project Overview
This project is a **full-scale AI trading platform** powered by:

- **Multi-Agent Reinforcement Learning (MARL)**
- **Stable-Baselines3 (PPO)**
- **PettingZoo Parallel Environments**
- **LSTM Price Forecasting**
- **Risk Engine (VaR, CVaR)**
- **Interactive Streamlit Dashboard**
- **Advanced Visualizations (candlestick, trade markers, replay slider, heatmaps)**

Each agent has a different trading personality:

| Agent | Trading Style |
|-------|----------------|
| 🟦 Conservative | Low-risk, minimal drawdowns |
| 🔴 Aggressive | High-risk, high-reward |
| 🟩 Momentum | Trend follower |
| 🟣 Mean-Reversion | RSI-based reversal logic |

All 4 agents are trained **independently**, then compared together in a **trading battle arena**.

---

## 🧠 Features

### ✔️ **Multi-Agent Reinforcement Learning**
- Custom PettingZoo environment  
- Real stock market data  
- PPO training for each agent  

### ✔️ **Advanced Trading Dashboard (Streamlit)**
- Candlestick chart  
- Buy/Sell markers  
- Portfolio vs. Market curve  
- Multi-Agent Battle  
- Interactive Replay Slider  
- Real-time Playback Animation  
- Downloadable Trading Journal (CSV/Excel)  
- Custom dark/neon cyber UI theme  

### ✔️ **Financial Risk Analysis**
- Value-at-Risk (VaR)
- Conditional VaR (CVaR)
- Return Heatmaps

### ✔️ **LSTM Price Forecasting**
Predict next 30 days using LSTM:
- MinMax scaling  
- 60-step window  
- PyTorch LSTM  
- Overlay with historical prices  

### ✔️ **Multi-Asset Support**
Choose from:
- AAPL  
- TSLA  
- BTC-USD  

### ✔️ **Audio Alerts**
- 🔊 Buy signals  
- 🔊 Sell signals  

---

## 🏗️ Project Structure

```

AI-Trading-Dashboard/
│
├── app.py                     # Main Streamlit dashboard
├── your_functions.py          # All backend functions & environment code
├── requirements.txt           # All dependencies
├── README.md                  # Project documentation
│
├── models/                    # Trained PPO models
│   ├── conservative_ppo_model.zip
│   ├── aggressive_ppo_model.zip
│   ├── momentum_ppo_model.zip
│   ├── mean_reversion_ppo_model.zip
│
├── assets/                    # Logo, banner, sound alerts
│   ├── logo.png
│   ├── banner_dark.png
│   ├── buy_alert.wav
│   ├── sell_alert.wav
│
├── data/                      # Optional cached market data
│   ├── cached_AAPL.csv
│   ├── cached_TSLA.csv
│   ├── cached_BTC-USD.csv
│
└── .streamlit/
├── config.toml            # Streamlit theme
└── theme.toml

````

---

## 📥 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/AI-Trading-Dashboard.git
cd AI-Trading-Dashboard
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📊 Training Environment

All RL agents are trained using:

* **Gymnasium + PettingZoo ParallelEnv**
* **Supersuit preprocessing**
* **Stable-Baselines3 PPO**
* Feature-rich observation space:

  ```
  [balance, holdings, close, RSI, SMA_20, MACD]
  ```

To retrain all agents:

```python
from your_functions import train_all_agents
train_all_agents()
```

---

## 📡 Data Source

Market data is fetched using **Yahoo Finance (yfinance)**:

* OHLCV data
* Technical indicators: RSI, SMA, MACD
* Multi-asset support

---

## 🧮 Risk Metrics

### **Value at Risk (VaR)**

Probability of the worst loss at 95% confidence.

### **Conditional VaR**

Expected loss beyond VaR threshold.

Displayed as:

* Numerical values
* Heatmap correlations

---

## 🔮 LSTM Forecasting

Predicts the next **30 days of closing price**:

* PyTorch LSTM
* 60-day training window
* MinMax scaling
* Real vs. forecast visualization

---

## 🎥 Interactive Visualization Features

### 🌟 Candlestick Chart

With neon-styled buy/sell markers.

### 🎚 Replay Slider

Manually scrub through the trading timeline.

### 🎥 Full Animation

Automatic playback of market vs. agent portfolio.

### ⚔️ Battle Arena

All 4 agents compete simultaneously.

### 📘 Trading Journal Export

CSV & Excel formats.

---

## 🌐 Deployment Options

### ✔ Streamlit Cloud — easiest

### ✔ HuggingFace Spaces

### ✔ Docker

### ✔ Render

### ✔ Localhost

---

## 🧩 requirements.txt

```
streamlit
pandas
numpy
plotly
matplotlib
seaborn
yfinance
ta
scikit-learn
gymnasium
pettingzoo
stable-baselines3
torch
torchaudio
torchvision
openpyxl
xlrd
```


---

## 🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss what you'd like to contribute.

---

## 📜 License

MIT License

---

## ✨ Author

**Sarvesh Roy**
AI/ML Engineer | Reinforcement Learning | Deep Learning | Data Science

