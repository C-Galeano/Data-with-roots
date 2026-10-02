"""Reinforcement Learning: Q-Learning agent with SGDRegressor."""

import random

import numpy as np
from sklearn import config_context
from sklearn.linear_model import SGDRegressor

from models.environment import ACTION_NAMES, ACTIONS, GridEnvironment

NUMBER_OF_ACTIONS = len(ACTIONS)

# Training configuration.
EPISODES = 300
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98
LEARNING_RATE = 0.1
SEED = 42

PARAMETERS = [
    {"name": "Training episodes", "symbol": "N", "value": EPISODES,
     "reason": "Enough to explore the 80 reachable cells; training stays under a minute on Render."},
    {"name": "Discount factor", "symbol": "γ", "value": GAMMA,
     "reason": "Values the +100 at T from far away, so the agent plans the whole route."},
    {"name": "Initial epsilon", "symbol": "ε₀", "value": EPSILON_START,
     "reason": "The agent knows nothing at first, so every action is random."},
    {"name": "Minimum epsilon", "symbol": "ε_min", "value": EPSILON_MIN,
     "reason": "Keeps 5% exploration so the agent can still correct bad estimates."},
    {"name": "Epsilon decay", "symbol": "decay", "value": EPSILON_DECAY,
     "reason": "ε is multiplied by 0.98 after each episode, reaching ε_min after ~150, half the run."},
    {"name": "Learning rate", "symbol": "η", "value": LEARNING_RATE,
     "reason": "Step size of partial_fit(): each update moves Q 10% toward the target."},
]


class QLearningAgent:
    """Q(s, a) approximated by an incrementally trained SGDRegressor."""

    def __init__(self, env, learning_rate=LEARNING_RATE, seed=SEED):
        self.env = env
        self.number_of_features = env.rows * env.columns * NUMBER_OF_ACTIONS
        self.model = SGDRegressor(
            loss="squared_error",
            penalty=None,
            fit_intercept=False,
            learning_rate="constant",
            eta0=learning_rate,
            random_state=seed,
        )
        # Cached Q-values per state.
        self.cache = {}
        # predict() fails until the model has seen one sample.
        self.model.partial_fit(
            np.zeros((1, self.number_of_features)),
            np.array([0.0]),
        )

    def encode(self, state, action):
        """One-hot vector with a single 1 for this state-action pair."""
        features = np.zeros(self.number_of_features)
        index = self.env.state_index(state) * NUMBER_OF_ACTIONS + action
        features[index] = 1.0
        return features

    def q_values(self, state):
        """Estimated Q-value of every action in this state."""
        if state not in self.cache:
            features = np.array([
                self.encode(state, action) for action in range(NUMBER_OF_ACTIONS)
            ])
            self.cache[state] = self.model.predict(features)
        return self.cache[state]

    def best_action(self, state, rng=None):
        """Greedy action; ties are broken at random during training."""
        q_values = self.q_values(state)
        best = np.flatnonzero(q_values == q_values.max())
        if rng is None:
            return int(best[0])
        return int(rng.choice(best.tolist()))

    def choose_action(self, state, epsilon, rng):
        """Epsilon-greedy: explore with probability epsilon, otherwise exploit."""
        if rng.random() < epsilon:
            return rng.randrange(NUMBER_OF_ACTIONS), "Exploration"
        return self.best_action(state, rng), "Exploitation"

    def update(self, state, action, reward, next_state, terminated, gamma):
        """Move Q(s, a) toward r + γ · max Q(s', a')."""
        if terminated:
            # T ends the episode, so there is no future reward.
            target = float(reward)
        else:
            target = reward + gamma * float(self.q_values(next_state).max())
        features = self.encode(state, action).reshape(1, -1)
        self.model.partial_fit(features, np.array([target]))
        # This state's Q-values changed.
        self.cache.pop(state, None)


def run_episode(env, agent, epsilon, gamma, rng):
    """Play one training episode and learn from each transition."""
    state = env.reset()
    total_reward = 0
    explored = 0
    while True:
        action, mode = agent.choose_action(state, epsilon, rng)
        explored += mode == "Exploration"
        result = env.step(action)
        agent.update(state, action, result["reward"], result["next_state"],
                     result["terminated"], gamma)
        total_reward += result["reward"]
        state = result["next_state"]
        if result["done"]:
            return {
                "reward": total_reward,
                "steps": result["step"],
                "success": result["terminated"],
                "explored": explored,
            }


def evaluate(env, agent):
    """Follow the learned policy with no exploration and no learning."""
    state = env.reset()
    path = [state]
    steps = []
    while True:
        action = agent.best_action(state)
        result = env.step(action)
        steps.append(result)
        state = result["next_state"]
        path.append(state)
        if result["done"]:
            break
    return {
        "steps": steps,
        "path": path,
        "reached_goal": steps[-1]["terminated"],
        "moves": len(steps),
        "total_reward": sum(step["reward"] for step in steps),
        "danger_cells": sum(step["cell_type"] == "Danger" for step in steps),
    }


def q_table(env, agent):
    """Q-values for every valid state."""
    table = []
    for state in env.valid_states():
        if state == env.goal:
            continue
        values = agent.q_values(state)
        table.append({
            "state": state,
            "cell": env.cell(state),
            "values": [round(float(value), 2) for value in values],
            "best_action": ACTION_NAMES[int(np.argmax(values))],
        })
    return table


def train(episodes=EPISODES, gamma=GAMMA, epsilon_start=EPSILON_START,
          epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, seed=SEED):
    """Train the agent and evaluate the learned policy."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    rng = random.Random(seed)
    env = GridEnvironment()
    agent = QLearningAgent(env, seed=seed)
    epsilon = epsilon_start
    history = []

    # Skip sklearn input checks to train faster.
    with config_context(assume_finite=True, skip_parameter_validation=True):
        for number in range(1, episodes + 1):
            episode = run_episode(env, agent, epsilon, gamma, rng)
            episode["number"] = number
            episode["epsilon"] = round(epsilon, 4)
            history.append(episode)
            # Explore less as the estimates improve.
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    evaluation = evaluate(env, agent)
    successes = sum(episode["success"] for episode in history)
    rewards = [episode["reward"] for episode in history]
    last = rewards[-100:]
    path_cells = set(evaluation["path"])

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(100 * successes / episodes, 1),
        "average_reward": round(sum(rewards) / episodes, 2),
        "last_100_average": round(sum(last) / len(last), 2),
        "final_epsilon": round(epsilon, 4),
        "evaluation": evaluation,
        "path": evaluation["path"],
        "grid": [
            [{"symbol": symbol, "on_path": (row, column) in path_cells}
             for column, symbol in enumerate(cells)]
            for row, cells in enumerate(env.grid)
        ],
        "q_table": q_table(env, agent),
        "history": history,
        "action_names": ACTION_NAMES,
    }
