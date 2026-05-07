from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from ... import config
import random

INITIAL_OFFSET = 5
SPAWN_INTERVAL = 4

class BasicZombieSpawner(ZombieSpawner):

    def __init__(self):
        self._timer = max(0.0, INITIAL_OFFSET - config.SIMULATION_DT)
        
    def spawn(self, scene):
        if self._timer <= 0:
            lane = random.choice(range(config.N_LANES))
            scene.add_zombie(Zombie(lane))
            self._timer = max(0.0, SPAWN_INTERVAL - config.SIMULATION_DT)
        else:
            self._timer -= config.SIMULATION_DT