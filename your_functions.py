import numpy as np
import pandas as pd
import yfinance as yf
import os

import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD

from stable_baselines3 import PPO


# =========================================================
# 1️⃣ MARKET DATA + INDICATORS (ROBUST)
# =========================================================
def get_market_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["MACD"] = MACD(close=df["Close"]).macd()

    df.dropna(inplace=True)

    if len(df) < 50:
        raise ValueError(f"Not enough processed data for {ticker}")

    df.reset_index(inplace=True)
    return df


# =========================================================
# 2️⃣ MULTI-AGENT TRADING ENVIRONMENT
# =========================================================
class StockTradingEnv(ParallelEnv):

    metadata = {"render_modes": ["human"], "name": "stock_trading_v0"}

    def __init__(self, df, initial_balance=10000):
        super().__init__()

        if df is None or len(df) == 0:
            raise ValueError("Empty dataframe passed to environment")

        self.df = df
        self.initial_balance = initial_balance

        self.possible_agents = [
            "conservative",
            "aggressive",
            "momentum",
            "mean_reversion",
        ]

        self.agents = self.possible_agents[:]

        self.action_spaces = {
            agent: spaces.Discrete(3)
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Box(
                low=-1e10,
                high=1e10,
                shape=(6,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    # ------------------------------
    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents[:]
        self.current_step = 0

        self.portfolio = {
            agent: {"balance": self.initial_balance, "holdings": 0}
            for agent in self.agents
        }

        return self._get_obs(), {agent: {} for agent in self.agents}

    # ------------------------------
    def step(self, actions):

        rewards = {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.current_step >= len(self.df) - 1:
            self.agents = []
            return {}, rewards, terminations, truncations, infos

        current_price = self.df.iloc[self.current_step]["Close"]

        for agent in self.agents:

            action = actions.get(agent, 0)

            balance = self.portfolio[agent]["balance"]
            holdings = self.portfolio[agent]["holdings"]

            prev_value = balance + holdings * current_price

            # BUY
            if action == 1 and balance >= current_price:
                self.portfolio[agent]["balance"] -= current_price
                self.portfolio[agent]["holdings"] += 1

            # SELL
            elif action == 2 and holdings > 0:
                self.portfolio[agent]["balance"] += current_price
                self.portfolio[agent]["holdings"] -= 1

            new_value = (
                self.portfolio[agent]["balance"]
                + self.portfolio[agent]["holdings"] * current_price
            )

            rewards[agent] = new_value - prev_value

        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            terminations = {agent: True for agent in self.agents}
            self.agents = []

        return self._get_obs(), rewards, terminations, truncations, infos

    # ------------------------------
    def _get_obs(self):

        obs = {}

        if self.current_step >= len(self.df):
            return obs

        row = self.df.iloc[self.current_step]

        features = [
            row["Close"],
            row["RSI"],
            row["SMA_20"],
            row["MACD"],
        ]

        for agent in self.agents:
            state = [
                self.portfolio[agent]["balance"],
                self.portfolio[agent]["holdings"],
            ] + features

            obs[agent] = np.array(state, dtype=np.float32)

        return obs


# =========================================================
# 3️⃣ SINGLE AGENT WRAPPER (FULLY SAFE)
# =========================================================
class SingleAgentWrapper(gym.Env):

    def __init__(self, df, agent_name):
        super().__init__()

        self.env = StockTradingEnv(df)
        self.agent = agent_name

        self.observation_space = self.env.observation_spaces[self.agent]
        self.action_space = self.env.action_spaces[self.agent]

    def reset(self, seed=None, options=None):

        obs, _ = self.env.reset(seed=seed)

        # SAFE RETURN
        if self.agent in obs:
            return obs[self.agent], {}
        else:
            return np.zeros(self.observation_space.shape), {}

    def step(self, action):

        if not self.env.agents:
            return np.zeros(self.observation_space.shape), 0, True, False, {}

        actions = {
            ag: (action if ag == self.agent else 0)
            for ag in self.env.agents
        }

        obs, rewards, terms, truncs, _ = self.env.step(actions)

        reward = rewards.get(self.agent, 0)
        terminated = terms.get(self.agent, True)
        truncated = truncs.get(self.agent, False)

        if self.agent in obs:
            next_obs = obs[self.agent]
        else:
            next_obs = np.zeros(self.observation_space.shape)
            terminated = True

        return next_obs, reward, terminated, truncated, {}


# =========================================================
# 4️⃣ PERFORMANCE METRICS
# =========================================================
def calculate_sharpe_ratio(returns):
    returns = np.array(returns)
    if len(returns) == 0 or returns.std() == 0:
        return 0
    return returns.mean() / returns.std()


def max_drawdown(values):
    if len(values) == 0:
        return 0
    values = np.array(values)
    peaks = np.maximum.accumulate(values)
    drawdown = (values - peaks) / peaks
    return drawdown.min()


def calculate_volatility(returns):
    return np.std(returns) if len(returns) else 0


def win_rate(rewards):
    rewards = np.array(rewards)
    return (rewards > 0).sum() / len(rewards) if len(rewards) else 0


# =========================================================
# 5️⃣ DETAILED EVALUATION
# =========================================================
def evaluate_agent_detailed(model, agent_name, df):

    env = SingleAgentWrapper(df, agent_name)
    obs, _ = env.reset()

    portfolio_values = []
    rewards_list = []
    buy_points = []
    sell_points = []

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        rewards_list.append(reward)
        done = terminated or truncated

        step = min(env.env.current_step, len(df) - 1)
        price = df.iloc[step]["Close"]

        port = env.env.portfolio.get(agent_name)
        if port:
            value = port["balance"] + port["holdings"] * price
            portfolio_values.append(value)

        if action == 1:
            buy_points.append((step, price))
        elif action == 2:
            sell_points.append((step, price))

    returns = np.diff(portfolio_values)

    metrics = {
        "Final Value": portfolio_values[-1] if portfolio_values else 0,
        "Sharpe Ratio": calculate_sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(portfolio_values),
        "Volatility": calculate_volatility(returns),
        "Win Rate": win_rate(rewards_list),
    }

    return portfolio_values, metrics, buy_points, sell_points


# =========================================================
# 6️⃣ MULTI-AGENT BATTLE (STREAMLIT SAFE)
# =========================================================
def multi_agent_battle(df, models):

    env = StockTradingEnv(df)
    obs, _ = env.reset()

    history = {agent: [] for agent in env.possible_agents}

    while env.agents:

        actions = {}

        for agent in env.agents:
            if agent in models:
                action, _ = models[agent].predict(obs[agent], deterministic=True)
                actions[agent] = action
            else:
                actions[agent] = 0

        obs, _, _, _, _ = env.step(actions)

        if env.current_step < len(df):
            price = df.iloc[env.current_step]["Close"]

            for agent in history:
                port = env.portfolio.get(agent)
                if port:
                    value = port["balance"] + port["holdings"] * price
                    history[agent].append(value)

    return history
