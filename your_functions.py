import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
import gymnasium as gym

# ================================================
# 1. DATA LOADING + INDICATORS
# ================================================
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD

def get_market_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["MACD"] = MACD(close=df["Close"]).macd()

    df.dropna(inplace=True)
    return df


# ================================================
# 2. MULTI-AGENT ENVIRONMENT (YOUR ORIGINAL ENV)
# ================================================
from pettingzoo import ParallelEnv

class StockTradingEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "stock_trading_v0"}

    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.agents = ["conservative", "aggressive", "momentum", "mean_reversion"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.agents}

        self.observation_spaces = {
            agent: spaces.Box(low=-1e10, high=1e10, shape=(6,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.portfolio = {
            agent: {"balance": self.initial_balance, "holdings": 0}
            for agent in self.agents
        }
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        rewards = {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        current_price = self.df.iloc[self.current_step]["Close"]

        for agent, action in actions.items():
            balance = self.portfolio[agent]["balance"]
            holdings = self.portfolio[agent]["holdings"]
            prev_val = balance + holdings * current_price

            # Buy
            if action == 1 and balance >= current_price:
                self.portfolio[agent]["balance"] -= current_price
                self.portfolio[agent]["holdings"] += 1

            # Sell
            elif action == 2 and holdings > 0:
                self.portfolio[agent]["balance"] += current_price
                self.portfolio[agent]["holdings"] -= 1

            # Reward shaping
            new_val = (
                self.portfolio[agent]["balance"]
                + self.portfolio[agent]["holdings"] * current_price
            )
            profit = new_val - prev_val

            if agent == "conservative":
                reward = profit if profit > 0 else profit * 2
            elif agent == "aggressive":
                reward = profit
            elif agent == "momentum":
                sma = self.df.iloc[self.current_step]["SMA_20"]
                reward = profit + (1 if action == 1 and current_price > sma else 0)
            elif agent == "mean_reversion":
                rsi = self.df.iloc[self.current_step]["RSI"]
                reward = (
                    profit + 2
                    if (action == 1 and rsi < 30) or (action == 2 and rsi > 70)
                    else profit
                )

            rewards[agent] = reward

        self.current_step += 1
        observations = self._get_obs()

        if self.current_step >= len(self.df) - 1:
            terminations = {agent: True for agent in self.agents}
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self):
        obs = {}
        if self.current_step < len(self.df):
            row = self.df.iloc[self.current_step]
            data_features = [row["Close"], row["RSI"], row["SMA_20"], row["MACD"]]

            for agent in self.agents:
                state = [
                    self.portfolio[agent]["balance"],
                    self.portfolio[agent]["holdings"],
                ] + data_features

                obs[agent] = np.array(state, dtype=np.float32)
        return obs


# ================================================
# 3. SINGLE AGENT WRAPPER (FOR PPO TRAINING)
# ================================================
class SingleAgentWrapper(gym.Env):
    def __init__(self, df, agent_name, initial_balance=10000):
        super().__init__()
        self.env = StockTradingEnv(df, initial_balance)
        self.agent = agent_name
        self.current_obs = None

        self.observation_space = self.env.observation_spaces[self.agent]
        self.action_space = self.env.action_spaces[self.agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed)
        self.current_obs = obs[self.agent]
        return self.current_obs, {}

    def step(self, action):
        actions = {ag: (action if ag == self.agent else 0) for ag in self.env.agents}
        next_obs, rewards, terms, truncs, infos = self.env.step(actions)

        terminated = terms[self.agent]
        truncated = False

        reward = rewards[self.agent]

        if self.agent in next_obs:
            self.current_obs = next_obs[self.agent]
        else:
            terminated = True

        return self.current_obs, reward, terminated, truncated, {}


# ================================================
# 4. PERFORMANCE METRICS
# ================================================
def calculate_sharpe_ratio(returns):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0
    return returns.mean() / returns.std()

def max_drawdown(values):
    values = np.array(values)
    peaks = np.maximum.accumulate(values)
    drawdown = (values - peaks) / peaks
    return drawdown.min()

def calculate_volatility(returns):
    return np.std(returns)

def win_rate(rewards):
    rewards = np.array(rewards)
    wins = (rewards > 0).sum()
    return wins / len(rewards) if len(rewards) > 0 else 0


# ================================================
# 5. DETAILED EVALUATION (METRICS + TRADES)
# ================================================
def evaluate_agent_detailed(model, agent_name, df):
    env = SingleAgentWrapper(df, agent_name)
    obs, _ = env.reset()

    done = False
    portfolio_values = []
    rewards_list = []
    buy_points = []
    sell_points = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards_list.append(reward)
        done = terminated or truncated

        price = df.iloc[env.env.current_step]["Close"]
        port = env.env.portfolio[agent_name]
        value = port["balance"] + port["holdings"] * price
        portfolio_values.append(value)

        if action == 1:
            buy_points.append((env.env.current_step, price))
        elif action == 2:
            sell_points.append((env.env.current_step, price))

    returns = np.diff(portfolio_values)

    metrics = {
        "Final Value": portfolio_values[-1],
        "Sharpe Ratio": calculate_sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(portfolio_values),
        "Volatility": calculate_volatility(returns),
        "Win Rate": win_rate(rewards_list),
    }

    return portfolio_values, metrics, buy_points, sell_points


# ================================================
# 6. MULTI-AGENT BATTLE SIMULATION
# ================================================
from stable_baselines3 import PPO

def multi_agent_battle(df):
    env = StockTradingEnv(df)
    obs, infos = env.reset()

    models = {
        "conservative": PPO.load("conservative_ppo_model"),
        "aggressive": PPO.load("aggressive_ppo_model"),
        "momentum": PPO.load("momentum_ppo_model"),
        "mean_reversion": PPO.load("mean_reversion_ppo_model"),
    }

    history = {agent: [] for agent in env.agents}

    while env.agents:
        actions = {}
        for agent in env.agents:
            model = models[agent]
            action, _ = model.predict(obs[agent], deterministic=True)
            actions[agent] = action

        obs, rewards, terms, truncs, infos = env.step(actions)

        if env.current_step < len(df):
            price = df.iloc[env.current_step]["Close"]
            for agent in history:
                port = env.portfolio[agent]
                value = port["balance"] + port["holdings"] * price
                history[agent].append(value)

    return history
