"""
Heuristic-based agent for Plants vs Zombies using rule-based strategies.

This agent implements a greedy strategy that evaluates plant placements
based on game state features like:
- Zombie threats (proximity, health, number)
- Sun availability
- Lane vulnerability
- Plant cooldowns
"""

import numpy as np
from pvz import config


class HeuristicAgent:
    """
    Rule-based heuristic agent that makes decisions based on game state.
    
    Strategy prioritization:
    1. Emergency defense: Place wall-nuts/chompers on lanes with closest zombies
    2. Sun generation: Place sunflowers on safe lanes early
    3. Offensive pressure: Place peashooters/repeaters on high-threat lanes
    4. Crowd control: Use jalapenos for zombie clusters
    """

    def __init__(self, possible_actions=None, n_plants=6):
        """
        Initialize the heuristic agent.
        
        Args:
            possible_actions: List of valid action indices (optional)
            n_plants: Number of different plant types available
        """
        self.possible_actions = possible_actions
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.n_plants = n_plants
        self.plant_costs = np.array([50, 100, 50, 150, 200, 125], dtype=np.float32)[:n_plants]
        self.plant_names = ["sunflower", "peashooter", "wall-nut", "chomper", "repeater", "jalapeno"][:n_plants]
        
        # Plant indices (matches the deck order in the environment)
        self.SUNFLOWER = 0
        self.PEASHOOTER = 1
        self.WALLNUT = 2
        self.CHOMPER = 3
        self.REPEATER = 4
        self.JALAPENO = 5

    def decide_action(self, observation, mask=None, epsilon=0, deterministic=False):
        """
        Decide the next action based on heuristic strategy.
        
        Observation layout:
        - [0:grid_size]: Plant grid (which plant is on each cell)
        - [grid_size:2*grid_size]: Zombie HP grid
        - [2*grid_size]: Current sun
        - [2*grid_size+1:]: Available plants and other info
        
        Args:
            observation: Game observation from the environment
            mask: Boolean mask of valid actions (optional)
            epsilon: Exploration rate (unused for heuristic)
            
        Returns:
            Action index to take
        """
        observation = np.asarray(observation, dtype=np.float32)
        state = self._parse_observation(observation)

        if mask is not None:
            available_actions = np.nonzero(mask)[0]
        else:
            available_actions = np.nonzero(self._get_mask(observation))[0]

        if len(available_actions) == 0:
            return 0

        best_action = 0
        best_score = -np.inf

        for action in available_actions:
            score = self._score_action(int(action), state)
            if score > best_score:
                best_score = score
                best_action = int(action)

        if mask is not None:
            return best_action, 0.0, 0.0

        return best_action

    def _parse_observation(self, observation):
        base_length = 2 * self._grid_size + 1 + self.n_plants
        grid = observation[:self._grid_size]
        plant_grid = grid.reshape(config.N_LANES, config.LANE_LENGTH)
        zombie_grid = observation[self._grid_size:2 * self._grid_size].reshape(config.N_LANES, config.LANE_LENGTH)

        state = {
            "plant_grid": plant_grid,
            "zombie_grid": zombie_grid,
            "sun": float(observation[2 * self._grid_size]),
            "action_available": observation[2 * self._grid_size + 1:2 * self._grid_size + 1 + self.n_plants].astype(bool),
            "cooldowns": np.zeros(self.n_plants, dtype=np.float32),
            "mowers": np.zeros(config.N_LANES, dtype=np.float32),
            "lane_zombie_hp": np.sum(zombie_grid, axis=1).astype(np.float32),
            "lane_plant_hp": np.count_nonzero(plant_grid, axis=1).astype(np.float32),
            "nearest_zombie": np.ones(config.N_LANES, dtype=np.float32),
            "global_features": np.zeros(4, dtype=np.float32),
            "wave_progress": 0.0,
            "wave_limit_progress": 0.0,
            "chrono_progress": 0.0,
        }

        if len(observation) > base_length:
            cursor = base_length
            state["cooldowns"] = observation[cursor:cursor + self.n_plants].astype(np.float32)
            cursor += self.n_plants
            state["mowers"] = observation[cursor:cursor + config.N_LANES].astype(np.float32)
            cursor += config.N_LANES
            state["lane_zombie_hp"] = observation[cursor:cursor + config.N_LANES].astype(np.float32)
            cursor += config.N_LANES
            state["lane_plant_hp"] = observation[cursor:cursor + config.N_LANES].astype(np.float32)
            cursor += config.N_LANES
            state["nearest_zombie"] = observation[cursor:cursor + config.N_LANES].astype(np.float32)
            cursor += config.N_LANES
            state["global_features"] = observation[cursor:].astype(np.float32)
            if len(state["global_features"]) > 0:
                state["wave_progress"] = float(state["global_features"][0])
            if len(state["global_features"]) > 1:
                state["wave_limit_progress"] = float(state["global_features"][1])
            if len(state["global_features"]) > 2:
                state["chrono_progress"] = float(state["global_features"][2])

        return state

    def _decode_action(self, action):
        action -= 1
        plant_type = action % self.n_plants
        cell_index = action // self.n_plants
        lane = cell_index % config.N_LANES
        pos = cell_index // config.N_LANES
        return lane, pos, plant_type

    def _lane_frontier(self, zombie_grid):
        closest_positions = np.full(config.N_LANES, -1, dtype=np.int32)
        occupied_counts = np.zeros(config.N_LANES, dtype=np.float32)

        for lane in range(config.N_LANES):
            lane_zombies = zombie_grid[lane]
            occupied = np.flatnonzero(lane_zombies > 0)
            occupied_counts[lane] = float(len(occupied))
            if len(occupied) > 0:
                closest_positions[lane] = int(occupied.max())

        return closest_positions, occupied_counts

    def _score_action(self, action, state):
        if action == 0:
            return -1e6

        lane, pos, plant_type = self._decode_action(action)
        zombie_grid = state["zombie_grid"]
        sun = state["sun"]
        wave_progress = state["wave_progress"]
        lane_plant_hp = state["lane_plant_hp"]

        closest_positions, occupied_counts = self._lane_frontier(zombie_grid)
        closest_pos = closest_positions[lane]
        lane_threat = self._analyze_lane_threats(zombie_grid)[lane]
        front_bias = pos / max(1, config.LANE_LENGTH - 1)

        plant_name = self.plant_names[plant_type]
        score = 0.0

        if plant_name == "sunflower":
            safe_lane = 1.0 if occupied_counts[lane] == 0 else max(0.0, 1.0 - lane_threat)
            score += 160.0
            score += 120.0 * safe_lane
            score += 60.0 * (1.0 - wave_progress)
            score += 20.0 if sun < 150 else 0.0
            score -= 80.0 * lane_threat
        elif plant_name == "peashooter":
            score += 110.0
            score += 140.0 * lane_threat
            score += 35.0 * occupied_counts[lane]
            score += 30.0 * wave_progress
            score += 20.0 * front_bias
        elif plant_name == "repeater":
            score += 95.0
            score += 170.0 * lane_threat
            score += 45.0 * occupied_counts[lane]
            score += 45.0 * wave_progress
            score += 25.0 * front_bias
        elif plant_name == "wall-nut":
            score += 130.0
            score += 180.0 * lane_threat
            if closest_pos >= 0:
                score += 120.0 if closest_pos >= config.LANE_LENGTH - 3 else 40.0
            score += 15.0 * front_bias
        elif plant_name == "chomper":
            score += 145.0
            score += 200.0 * lane_threat
            if closest_pos >= 0:
                score += 160.0 if closest_pos >= config.LANE_LENGTH - 3 else 50.0
            score += 20.0 * wave_progress
            score += 20.0 * front_bias
        elif plant_name == "jalapeno":
            emergency = 0.0
            if closest_pos >= 0:
                emergency = max(0.0, 1.0 - (config.LANE_LENGTH - closest_pos) / 3.0)
            score += 40.0 + 260.0 * emergency
            score += 120.0 * lane_threat
            score += 40.0 * occupied_counts[lane]
            score += 25.0 * wave_progress
            score += 20.0 * front_bias

        if closest_pos >= 0 and plant_name in {"jalapeno", "wall-nut", "chomper"}:
            score += 75.0 if closest_pos >= config.LANE_LENGTH - 2 else 0.0

        if plant_name in {"peashooter", "repeater"} and occupied_counts[lane] > 0:
            score += 35.0

        if plant_name == "sunflower" and occupied_counts[lane] == 0:
            score += 45.0

        score += min(float(lane_plant_hp[lane]) * 5.0, 20.0)

        return score

    def _get_mask(self, observation):
        """Generate valid action mask from observation."""
        grid = observation[:self._grid_size]
        empty_cells = np.nonzero((grid == 0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(1 + self._grid_size * self.n_plants, dtype=bool)
        mask[0] = True  # Do nothing action
        
        empty_cells_flat = empty_cells[0] + config.N_LANES * empty_cells[1]
        sun_available = observation[2 * self._grid_size]
        
        # Mark available plant placements
        available_plants = observation[-self.n_plants:].astype(bool)
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells_flat * self.n_plants + i + 1
                mask[idx] = True
        
        return mask

    def _analyze_lane_threats(self, zombie_grid):
        """
        Analyze threat level for each lane.
        
        Returns array of threat scores per lane.
        Higher score = higher threat.
        """
        threat = np.zeros(config.N_LANES)
        
        for lane in range(config.N_LANES):
            lane_zombies = zombie_grid[lane]
            
            # Find closest zombie in this lane
            closest_zombie = None
            min_distance = float('inf')
            for pos in range(config.LANE_LENGTH):
                if lane_zombies[pos] > 0:
                    distance = config.LANE_LENGTH - pos
                    if distance < min_distance:
                        min_distance = distance
                        closest_zombie = pos
            
            if closest_zombie is not None:
                # Threat = zombie HP / distance (closer/stronger = higher threat)
                distance_factor = max(1, config.LANE_LENGTH - closest_zombie)
                total_hp = np.sum(lane_zombies)
                threat[lane] = total_hp / (1.0 + distance_factor)
        
        return threat

    def _select_action_by_priority(self, plant_grid, threat_by_lane, sun, available_actions):
        """
        Select action based on priority heuristics.
        
        Priority order:
        1. Emergency defense on high-threat lanes
        2. Sun generation on safe lanes
        3. Offensive plants on moderate-threat lanes
        4. Crowd control with jalapenos
        """
        # Strategy 1: High threat lanes need defense
        high_threat_lanes = np.where(threat_by_lane > 3.0)[0]
        if len(high_threat_lanes) > 0:
            # Try to place defensive plants (wallnut, chomper)
            action = self._try_place_plants(
                plant_grid, 
                high_threat_lanes, 
                [self.WALLNUT, self.CHOMPER],
                available_actions,
                min_sun=50
            )
            if action is not None:
                return action
        
        # Strategy 2: Medium threat lanes need offense
        medium_threat_lanes = np.where((threat_by_lane > 1.0) & (threat_by_lane <= 3.0))[0]
        if len(medium_threat_lanes) > 0 and sun >= 100:
            action = self._try_place_plants(
                plant_grid,
                medium_threat_lanes,
                [self.REPEATER, self.PEASHOOTER],
                available_actions,
                min_sun=100
            )
            if action is not None:
                return action
        
        # Strategy 3: Low threat lanes - generate sun
        low_threat_lanes = np.where(threat_by_lane < 1.0)[0]
        if len(low_threat_lanes) > 0 and sun < 150:
            action = self._try_place_plants(
                plant_grid,
                low_threat_lanes,
                [self.SUNFLOWER],
                available_actions,
                min_sun=50
            )
            if action is not None:
                return action
        
        # Strategy 4: Place any available plant if we have sun
        if sun >= 50:
            action = self._try_place_plants(
                plant_grid,
                np.arange(config.N_LANES),
                [self.PEASHOOTER, self.REPEATER, self.WALLNUT, self.SUNFLOWER],
                available_actions,
                min_sun=50
            )
            if action is not None:
                return action
        
        # Fallback: Do nothing
        return 0

    def _try_place_plants(self, plant_grid, lanes, plant_priorities, available_actions, min_sun):
        """
        Try to place plants from priority list in given lanes.
        
        Returns action index if successful placement found, None otherwise.
        """
        for lane in lanes:
            for plant_type in plant_priorities:
                # Find best empty position in lane (prefer front-back for defense)
                best_pos = self._find_best_empty_position(plant_grid, lane)
                
                if best_pos is not None:
                    # Convert to action index
                    action = self._cell_plant_to_action(lane, best_pos, plant_type)
                    
                    if action in available_actions:
                        return action
        
        return None

    def _find_best_empty_position(self, plant_grid, lane):
        """
        Find the best empty position in a lane.
        
        Prefer front positions (closer to zombies) for defense,
        back positions for sun generation.
        """
        lane_grid = plant_grid[lane]
        empty_positions = np.where(lane_grid == 0)[0]
        
        if len(empty_positions) == 0:
            return None
        
        # Prefer positions closer to zombies (front of lane)
        return int(empty_positions[-1])  # Rightmost empty position (furthest back)

    def _cell_plant_to_action(self, lane, pos, plant_type):
        """
        Convert (lane, position, plant_type) to action index.
        
        Action encoding: 0 = do nothing
        1 + plant_index + cell_index * n_plants
        where cell_index = lane + pos * n_lanes
        """
        cell_index = lane + pos * config.N_LANES
        action = 1 + plant_type + cell_index * self.n_plants
        return int(action)
