from ... import config
from .zombie import Zombie


class Zombie_all_star(Zombie):

    MAX_HP = 1600
    SCORE = 300
    WALKING_SPEED = 4
    ATTACK_SPEED = 300

    def __init__(self, lane, pos=config.LANE_LENGTH - 1):
        super().__init__(lane, pos)
        self._slowed_after_first_hit = False

    def step(self, scene):
        charge_threshold = 800

        if self._slowed_after_first_hit:
            self._cell_length = self.WALKING_SPEED
        elif self.hp > charge_threshold:
            self._cell_length = max(config.SIMULATION_DT, self.WALKING_SPEED / 2)
        else:
            self._cell_length = self.WALKING_SPEED

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
                    self._slowed_after_first_hit = True
                    break
