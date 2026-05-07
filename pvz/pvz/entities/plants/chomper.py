from .plant import Plant
from ... import config

CHOMPER_COST = 150
CHOMPER_COOLDOWN = 15
CHOMPER_MAX_HP = 300
CHOMPER_DIGEST_COOLDOWN = 16


class Chomper(Plant):

    MAX_HP = CHOMPER_MAX_HP
    COOLDOWN = CHOMPER_COOLDOWN
    COST = CHOMPER_COST
    DIGEST_COOLDOWN = CHOMPER_DIGEST_COOLDOWN

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.digest_cooldown = 0

    def step(self, scene):
        if self.digest_cooldown > 0:
            self.digest_cooldown -= config.SIMULATION_DT
            return

        for zombie in scene.zombies:
            if zombie.lane == self.lane and zombie.pos == self.pos:
                zombie.hp = 0
                self.digest_cooldown = max(0.0, self.DIGEST_COOLDOWN - config.SIMULATION_DT)
                break