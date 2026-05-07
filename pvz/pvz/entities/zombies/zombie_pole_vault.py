from ... import config
from .zombie import Zombie
from ..plants.potatomine import Potatomine
from ..plants.chomper import Chomper


class Zombie_pole_vault(Zombie):

    MAX_HP = 340
    SCORE = 170

    def __init__(self, lane, pos=config.LANE_LENGTH - 1):
        super().__init__(lane, pos)
        self._jumped = False

    def step(self, scene):
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
            if not self._jumped:
                # Jump once for any obstacle
                self._jumped = True
                self.pos -= 1
                self._offset = max(0.0, self._cell_length - config.SIMULATION_DT)
                if self.pos < 0:
                    scene.zombie_reach(self.lane)
                    self.hp = 0
            else:
                # After landing from jump, check if we're on a potato mine/chomper
                for plant in scene.plants:
                    if (plant.lane == self.lane) and (plant.pos == self.pos):
                        if isinstance(plant, (Potatomine, Chomper)):
                            # Jump over it instead of attacking
                            self._jumped = True
                            self.pos -= 1
                            self._offset = max(0.0, self._cell_length - config.SIMULATION_DT)
                            if self.pos < 0:
                                scene.zombie_reach(self.lane)
                                self.hp = 0
                        else:
                            # Attack other plants normally
                            self.attack(plant)
                        break