import numpy as np
from pvz import config


class RandomAgent:
    """Choose a random valid action based on the observation mask.

    Observation layout matches other agents: first `grid_size` plant cells,
    then zombie info, then sun, then available plants (length `n_plants`).
    Action encoding: `0` = do nothing, else `1 + cell_index * n_plants + plant_index`.
    """
    def __init__(self, possible_actions=None, n_plants=4):
        # possible_actions: list of valid action indices (preferred)
        self.possible_actions = possible_actions
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.n_plants = n_plants

    def _get_mask(self, observation):
        grid = observation[: self._grid_size]
        empty_cells = np.nonzero((grid == 0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(1 + self._grid_size * self.n_plants, dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * self.n_plants

        available_plants = observation[-self.n_plants :]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask

    def decide_action(self, observation, mask=None, epsilon=0):
        # Accept PlayerQ-style signature (observation, mask, epsilon)
        if self.possible_actions is not None:
            return int(np.random.choice(self.possible_actions))
        if mask is not None:
            candidates = np.nonzero(mask)[0]
            return int(np.random.choice(candidates))
        mask = self._get_mask(observation)
        candidates = np.nonzero(mask)[0]
        return int(np.random.choice(candidates))


class DoNothingAgent:
    """Always returns the 'do nothing' action (0). Useful as a baseline."""

    def decide_action(self, observation, mask=None, epsilon=0):
        return 0
