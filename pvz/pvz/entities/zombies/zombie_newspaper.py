from ... import config
from .zombie import Zombie


class Zombie_newspaper(Zombie):

    MAX_HP = 320
    SCORE = 180
    WALKING_SPEED = 3
    RAGE_SPEED = 7
    RAGE_THRESHOLD = 190

    def __init__(self, lane, pos=config.LANE_LENGTH - 1):
        super().__init__(lane, pos)
        self._raged = False

    def _rage(self):
        self._raged = True
        self._cell_length = self.RAGE_SPEED
        self._offset = min(self._offset, max(0.0, self._cell_length - config.SIMULATION_DT))

    def step(self, scene):
        if (not self._raged) and self.hp <= self.RAGE_THRESHOLD:
            self._rage()

        if scene.grid.is_empty(self.lane, self.pos):
            if self._offset <= 0:
                self.pos -= 1
                self._offset = max(0.0, self._cell_length - config.SIMULATION_DT)
                if self.pos < 0:
                    scene.zombie_reach(self.lane)
                    self.hp = 0
            else:
                self._offset -= config.SIMULATION_DT
        else:
            for plant in scene.plants:
                if (plant.lane == self.lane) and (plant.pos == self.pos):
                    self.attack(plant)
                    break

        if (not self._raged) and self.hp <= self.RAGE_THRESHOLD:
            self._rage()