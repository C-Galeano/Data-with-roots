"""
Reinforcement Learning: grid environment and reward system.

A robot starts at A and must reach T on a 10x10 grid. Walls (#) block the way,
danger zones (D) can be crossed but are heavily penalized, and every other
move costs a little so the agent prefers short routes. The Q-Learning agent
only talks to this module through reset() and step().
"""

from collections import Counter
import heapq

# A = start, T = target, o = path, # = wall, D = danger zone.
LAYOUT = [
    "A o o o o # o o o o",
    "o # o D o # o # # o",
    "o o # D o o o D # o",
    "o o o o # # D o o o",
    "o o # D o o o o # o",
    "o o # o o # D # o o",
    "o D o o # o o D o o",
    "o o # o o o # o o o",
    "o o o # D o D o # o",
    "o o o o o # o o o T",
]

START = "A"
TARGET = "T"
PATH = "o"
WALL = "#"
DANGER = "D"

CELL_TYPES = {
    START: "Start",
    TARGET: "Target",
    PATH: "Path",
    WALL: "Wall",
    DANGER: "Danger",
}

# Row and column changes for each action.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

# Every normal step costs 1, so shorter routes collect more reward.
REWARD_STEP = -1
# Trying to leave the grid wastes a step; the agent stays in place.
REWARD_OUT_OF_BOUNDS = -3
# Hitting a wall is a collision, worse than a harmless invalid move.
REWARD_WALL = -5
# Entering D costs 15 instead of 1, so any detour of up to 13 extra steps
# is worth taking. On this map the safe route is only 4 steps longer.
REWARD_DANGER = -15
# Larger than the cost of any reasonable route, so reaching T always pays.
REWARD_GOAL = 100

# The episode is cut off if the agent wanders for too long.
MAX_STEPS = 200

REWARD_TABLE = [
    {"event": "Move to a normal cell (o / A)", "reward": REWARD_STEP,
     "reason": "Small cost per step pushes the agent toward short routes."},
    {"event": "Invalid move (outside the grid)", "reward": REWARD_OUT_OF_BOUNDS,
     "reason": "The step is wasted and the agent stays in the same cell."},
    {"event": "Hit a wall (#)", "reward": REWARD_WALL,
     "reason": "A collision is worse than a wasted step; the agent stays in place."},
    {"event": "Enter a danger zone (D)", "reward": REWARD_DANGER,
     "reason": "Allowed but costly, so the agent learns to go around it."},
    {"event": "Reach the target (T)", "reward": REWARD_GOAL,
     "reason": "Ends the episode and outweighs the cost of the whole route."},
]


class GridEnvironment:
    """10x10 maze with walls, danger zones, and a single target."""

    def __init__(self, layout=LAYOUT, max_steps=MAX_STEPS):
        self.grid = [row.split() for row in layout]
        self.rows = len(self.grid)
        self.columns = len(self.grid[0])
        self.max_steps = max_steps
        self.start = self._find(START)
        self.goal = self._find(TARGET)
        self.state = self.start
        self.steps = 0

    def _find(self, symbol):
        for row in range(self.rows):
            for column in range(self.columns):
                if self.grid[row][column] == symbol:
                    return (row, column)
        raise ValueError(f"layout has no {symbol!r} cell")

    def cell(self, state):
        return self.grid[state[0]][state[1]]

    def in_bounds(self, row, column):
        return 0 <= row < self.rows and 0 <= column < self.columns

    def reset(self):
        """Put the agent back on A and start a new episode."""
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action):
        """Apply one action and describe what happened."""
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"action must be between 0 and {len(ACTIONS) - 1}")

        state = self.state
        row = state[0] + ACTIONS[action][0]
        column = state[1] + ACTIONS[action][1]

        # Blocked moves leave the agent where it was.
        if not self.in_bounds(row, column):
            next_state = state
            cell_type = "Out of bounds"
            reward = REWARD_OUT_OF_BOUNDS
        elif self.grid[row][column] == WALL:
            next_state = state
            cell_type = CELL_TYPES[WALL]
            reward = REWARD_WALL
        else:
            next_state = (row, column)
            symbol = self.grid[row][column]
            cell_type = CELL_TYPES[symbol]
            if symbol == TARGET:
                reward = REWARD_GOAL
            elif symbol == DANGER:
                reward = REWARD_DANGER
            else:
                reward = REWARD_STEP

        self.state = next_state
        self.steps += 1

        # terminated: reached T. truncated: ran out of steps.
        terminated = next_state == self.goal
        truncated = not terminated and self.steps >= self.max_steps

        return {
            "step": self.steps,
            "state": state,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
        }

    def valid_states(self):
        """Every cell the agent can stand on (everything except walls)."""
        return [
            (row, column)
            for row in range(self.rows)
            for column in range(self.columns)
            if self.grid[row][column] != WALL
        ]

    def state_index(self, state):
        return state[0] * self.columns + state[1]

    def cell_counts(self):
        counts = Counter(symbol for row in self.grid for symbol in row)
        return {symbol: counts.get(symbol, 0) for symbol in CELL_TYPES}

    def shortest_path(self, danger_cost=1):
        """
        Cheapest route from A to T, where entering D costs danger_cost moves.

        Used only to check the map design, never by the agent.
        """
        queue = [(0, self.start, [self.start])]
        visited = set()
        while queue:
            cost, state, path = heapq.heappop(queue)
            if state == self.goal:
                return path
            if state in visited:
                continue
            visited.add(state)
            for dr, dc in ACTIONS:
                row, column = state[0] + dr, state[1] + dc
                if not self.in_bounds(row, column):
                    continue
                symbol = self.grid[row][column]
                if symbol == WALL:
                    continue
                move_cost = danger_cost if symbol == DANGER else 1
                heapq.heappush(
                    queue,
                    (cost + move_cost, (row, column), path + [(row, column)]),
                )
        return None
