from .plant import Plant
from ... import config

SUNFLOWER_COST = 50
SUNFLOWER_COOLDOWN = 5
SUNFLOWER_MAX_HP = 300

class Sunflower(Plant):

    # Entity
    MAX_HP = SUNFLOWER_MAX_HP

    # Plant
    COOLDOWN = SUNFLOWER_COOLDOWN
    COST = SUNFLOWER_COST

    # Sunflower
    PRODUCTION = config.SUNFLOWER_PRODUCTION
    PRODUCTION_COOLDOWN = config.SUNFLOWER_PRODUCTION_COOLDOWN
    INITIAL_COOLDOWN = config.SUNFLOWER_INITIAL_COOLDOWN

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.production_cooldown = max(0.0, self.INITIAL_COOLDOWN - config.SIMULATION_DT)
    
    def step(self, scene):
        if self.production_cooldown <= 0:
            scene.sun += self.PRODUCTION
            self.production_cooldown = max(0.0, self.PRODUCTION_COOLDOWN - config.SIMULATION_DT)
        else:
            self.production_cooldown -= config.SIMULATION_DT
