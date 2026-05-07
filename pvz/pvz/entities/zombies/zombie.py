from ..entity import Entity

from ... import config

class Zombie(Entity):

    MAX_HP = 190  # 190 normally
    WALKING_SPEED = 5 # Speed per square
    ATTACK_SPEED = 100 # Damage per second
    SCORE = 120

    def __init__(self, lane, pos=config.LANE_LENGTH - 1): # Zombie starts at the very right
        """
        lane: lane where the zombie is
        """

        super().__init__(lane)
        self.pos = pos
        self._attack = max(1, int(self.ATTACK_SPEED * config.SIMULATION_DT))
        self._cell_length = self.WALKING_SPEED # Seconds to pass a cell
        self._offset = max(0.0, self._cell_length - config.SIMULATION_DT) # Remaining time before the zombie moves


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

    def attack(self, plant):
        plant.hp -= self._attack
    
    def get_offset(self):
        return self._offset/self._cell_length

    def __str__(self):
        return ("Lane: " + str(self.lane) + " Pos: " + str(self.pos)
                + " Health: " + str(self.hp))