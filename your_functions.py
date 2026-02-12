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
# 1️⃣ SAFE MARKET DATA LOADER
# =========================================================
def get_market_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["MACD"] = MACD(close=df["Close"]).macd()

    df.dropna(inplace=True)

    if len(df) < 50:
        raise ValueError(f"Not enough data after indicators for {ticker}")

    df.reset_index(inplace=True)

    return df


# =========================================================
# 2️⃣ MULTI-AGENT ENVIRONMENT (ROBUST VERSION)
# =========================================================
class StockTradingEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "stock_trading_v0"}

    def __init__(self, df, initial_balance=10000):
        super().__init__()

        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty in StockTradingEnv")

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
            agent: spaces.Discrete(3) for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Box(
                low=-1e10, high=1e10, shape=(6,), dtype=np.float32
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

        if self.current_step >= len(self.df):
            self.agents = []
            return {}, rewards, terminations, truncations, infos

        current_price = self.df.iloc[self.current_step]["Close"]

        for agent, action in actions.items():
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
        features = [row["Close"], row["RSI"], row["SMA_20"], row["MACD"]]

        for agent in self.agents:
            state = [
                self.portfolio[agent]["balance"],
                self.portfolio[agent]["holdings"],
            ] + features

            obs[agent] = np.array(state, dtype=np.float32)

        return obs


# =========================================================
# 3️⃣ SAFE SINGLE AGENT WRAPPER
# =========================================================
class SingleAgentWrapper(gym.Env):
    def __init__(self, df, agent_name):
        super().__init__()

        if agent_name not in [
            "conservative",
            "aggressive",
            "momentum",
            "mean_reversion",
        ]:
            raise ValueError(f"Invalid agent name: {agent_name}")

        self.env = StockTradingEnv(df)
        self.agent = agent_name

        self.observation_space = self.env.observation_spaces[self.agent]
        self.action_space = self.env.action_spaces[self.agent]

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)

        if not obs or self.agent not in obs:
            return np.zeros(self.observation_space.shape), {}

        return obs[self.agent], {}

    def step(self, action):
        actions = {
            ag: (action if ag == self.agent else 0)
            for ag in self.env.agents
        }

        obs, rewards, terms, truncs, _ = self.env.step(actions)

        terminated = terms.get(self.agent, True)
        truncated = truncs.get(self.agent, False)
        reward = rewards.get(self.agent, 0)

        if obs and self.agent in obs:
            next_obs = obs[self.agent]
        else:
            next_obs = np.zeros(self.observation_space.shape)
            terminated = True

        return next_obs, reward, terminated, truncated, {}
