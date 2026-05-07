from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from .zombie_cone import Zombie_cone
from .zombie_bucket import Zombie_bucket
from .zombie_flag import Zombie_flag
from .zombie_newspaper import Zombie_newspaper
from .zombie_pole_vault import Zombie_pole_vault
from .zombie_all_star import Zombie_all_star
from ... import config
import random

INITIAL_OFFSET = 6
SPAWN_INTERVAL = 8
REST_INTERVAL = 20
FIRST_HUGE_WAVE = 5
HUGE_WAVE_EVERY = 5
BASE_WAVE_BUDGET = 1
WAVE_BUDGET_SCALING = 1

ZOMBIE_CATALOG = (
    (Zombie, 1, lambda wave: 4000),
    (Zombie_cone, 2, lambda wave: min(1200 + wave * 80, 2800)),
    (Zombie_bucket, 4, lambda wave: min(150 + wave * 35, 1800)),
    (Zombie_newspaper, 3, lambda wave: min(220 + wave * 20, 650)),
    (Zombie_pole_vault, 2, lambda wave: min(180 + wave * 70, 900)),
    (Zombie_all_star, 6, lambda wave: min(50 + wave * 25, 2000)),
)

class WaveZombieSpawner(ZombieSpawner):

    def __init__(self):
        self._spawn_timer = max(0.0, INITIAL_OFFSET - config.SIMULATION_DT)
        self._wave_timer = 0
        self._wave_index = 1
        self._wave_budget = 0
        self._wave_hp = 0
        self._wave_active = False
        self._lane_cycle = []
        self._lane_cycle_index = 0
        self._flag_spawned_this_wave = False
        self._finished = False

    def _active_zombie_hp(self, scene):
        return sum(zombie.hp for zombie in scene.zombies)

    def _wave_budget_for(self, wave_index):
        base_budget = BASE_WAVE_BUDGET + wave_index * WAVE_BUDGET_SCALING
        if self._is_huge_wave(wave_index):
            return min(140, base_budget * 4)
        return min(90, base_budget)

    def _is_huge_wave(self, wave_index):
        if wave_index < FIRST_HUGE_WAVE:
            return False
        return (wave_index - FIRST_HUGE_WAVE) % HUGE_WAVE_EVERY == 0

    def _spawn_interval_for(self, wave_index):
        seconds = max(3, SPAWN_INTERVAL - wave_index // 6)
        return seconds

    def _rest_duration_for(self, wave_index):
        seconds = REST_INTERVAL + min(15, wave_index // 2)
        if self._is_huge_wave(wave_index):
            seconds += 5
        return seconds

    def _spawn_burst_size(self, wave_index):
        if self._is_huge_wave(wave_index):
            return 4
        if wave_index >= 12:
            return 3
        if wave_index >= 6:
            return 2
        return 1

    def _pool_for_wave(self, wave_index, remaining_budget):
        pool = []
        for zombie_class, cost, weight_fn in ZOMBIE_CATALOG:
            if cost <= remaining_budget:
                pool.append((zombie_class, cost, weight_fn(wave_index)))
        return pool

    def _pick_lane(self, huge_wave=False):
        if huge_wave:
            if self._lane_cycle_index >= len(self._lane_cycle):
                self._lane_cycle = list(range(config.N_LANES))
                random.shuffle(self._lane_cycle)
                self._lane_cycle_index = 0
            lane = self._lane_cycle[self._lane_cycle_index]
            self._lane_cycle_index += 1
            return lane
        return random.randrange(config.N_LANES)

    def _pick_zombie(self, wave_index, remaining_budget):
        pool = self._pool_for_wave(wave_index, remaining_budget)
        if not pool:
            return None
        weights = [entry[2] for entry in pool]
        chosen = random.choices(pool, weights=weights, k=1)[0]
        return chosen[0], chosen[1]

    def _begin_wave(self):
        self._wave_active = True
        self._wave_budget = self._wave_budget_for(self._wave_index)
        self._wave_hp = 0
        self._wave_timer = max(0.0, self._rest_duration_for(self._wave_index) - config.SIMULATION_DT)
        self._spawn_timer = 0
        self._lane_cycle = []
        self._lane_cycle_index = 0
        self._flag_spawned_this_wave = False

    def _finish_wave(self):
        self._wave_active = False
        self._spawn_timer = max(0.0, self._rest_duration_for(self._wave_index) - config.SIMULATION_DT)
        self._wave_index += 1

    def _spawn_wave_zombie(self, scene, huge_wave=False):
        picked = self._pick_zombie(self._wave_index, self._wave_budget)
        if picked is None:
            return False

        zombie_class, cost = picked
        lane = self._pick_lane(huge_wave=huge_wave)
        scene.add_zombie(zombie_class(lane))
        self._wave_budget -= cost
        self._wave_hp += getattr(zombie_class, "MAX_HP", Zombie.MAX_HP)
        return True

    def spawn(self, scene):
        active_hp = self._active_zombie_hp(scene)

        # Stop spawning once we've passed the configured maximum wave
        if self._wave_index > config.MAX_WAVE:
            self._finished = True
            return

        if self._wave_active and self._wave_timer > 0:
            self._wave_timer -= config.SIMULATION_DT

        if not self._wave_active:
            if self._spawn_timer > 0:
                self._spawn_timer -= config.SIMULATION_DT
                return

            self._begin_wave()
            # Spawn flag zombie at the start of huge waves.
            if self._is_huge_wave(self._wave_index) and not self._flag_spawned_this_wave:
                lane = self._pick_lane(huge_wave=True)
                scene.add_zombie(Zombie_flag(lane))
                self._wave_hp += Zombie_flag.MAX_HP
                self._flag_spawned_this_wave = True

        if self._wave_budget <= 0:
            if active_hp <= self._wave_hp / 2 or self._wave_timer <= 0:
                self._finish_wave()
            return

        if self._spawn_timer > 0:
            self._spawn_timer -= config.SIMULATION_DT
            return

        spawned = 0
        burst_size = self._spawn_burst_size(self._wave_index)
        
        # For huge waves, spawn on all lanes simultaneously
        if self._is_huge_wave(self._wave_index):
            for _ in range(burst_size):
                if self._wave_budget <= 0:
                    break
                for lane in range(config.N_LANES):
                    if self._wave_budget <= 0:
                        break
                    if not self._spawn_wave_zombie(scene, huge_wave=True):
                        return
        else:
            # Normal spawning - 1 lane per burst item
            while spawned < burst_size and self._wave_budget > 0:
                if not self._spawn_wave_zombie(scene, huge_wave=False):
                    break
                spawned += 1

        if self._wave_budget <= 0:
            self._finish_wave()
        else:
            self._spawn_timer = max(0.0, self._spawn_interval_for(self._wave_index) - config.SIMULATION_DT)