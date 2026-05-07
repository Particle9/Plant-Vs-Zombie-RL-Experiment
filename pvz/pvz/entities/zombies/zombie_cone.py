from ... import config
from .zombie import Zombie



class Zombie_cone(Zombie):

    MAX_HP = 560  
    SCORE = 40 # It will add 30 for the normal zombie kill

    def __init__(self, lane, pos=config.LANE_LENGTH - 1):
        super().__init__(lane, pos)
        self._shed_zombie = None

    def step(self, scene):
        if scene.grid.is_empty(self.lane, self.pos):
            if self._offset <= 0:
                self.pos -= 1
                self._offset = max(0.0, self._cell_length - config.SIMULATION_DT)
                if self.pos < 0: # If the zombie reached the house, we lose a life and the zombie disappear
                    scene.zombie_reach(self.lane)
                    self.hp = 0
            else:
                self._offset -= config.SIMULATION_DT
        else:
            for plant in scene.plants:
                if (plant.lane == self.lane) and (plant.pos == self.pos):
                    self.attack(plant)
                    break
        if self.hp<190 and self._shed_zombie is None:
            zombie = Zombie(self.lane,self.pos)
            zombie.hp = self.hp
            zombie._offset = self._offset
            self._shed_zombie = zombie
            self.hp=0
    
