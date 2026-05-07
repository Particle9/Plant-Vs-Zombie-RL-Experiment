from .plant import Plant
from ... import config


JALAPENO_COST = 125
JALAPENO_COOLDOWN = 50
JALAPENO_MAX_HP = 300
JALAPENO_ARMING_COOLDOWN = 0.5


class Jalapeno(Plant):

    MAX_HP = JALAPENO_MAX_HP
    COOLDOWN = JALAPENO_COOLDOWN
    COST = JALAPENO_COST
    ARMING_COOLDOWN = JALAPENO_ARMING_COOLDOWN

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.arming_cooldown = max(0.0, self.ARMING_COOLDOWN - config.SIMULATION_DT)
        self.used = False

    def step(self, scene):
        if self.used:
            self.hp = 0
            return

        if self.arming_cooldown > 0:
            self.arming_cooldown -= config.SIMULATION_DT
            return

        for zombie in scene.zombies:
            if zombie.lane == self.lane:
                zombie.hp = 0
        self.used = True
        self.hp = 0