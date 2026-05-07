FPS = 24

SIMULATION_SPEED = 10
SIMULATION_DT = SIMULATION_SPEED / FPS
 
# Maximum number of waves to run before stopping the level
MAX_WAVE = 100

# Backwards-compatible frame/second limits (used elsewhere)
MAX_FRAMES = 999999999
MAX_SECONDS = MAX_FRAMES / FPS

N_LANES = 5 # Height
LANE_LENGTH = 9 # Width

INITIAL_SUN_AMOUNT = 50

# Sun that falls from the sky
NATURAL_SUN_PRODUCTION = 25
NATURAL_SUN_PRODUCTION_COOLDOWN = 10

# Mowers
MOWERS = True

# Score values
SURVIVAL = 0
SURVIVAL_STEP = 20 # every 20 sec survival reward increases
SCORE_ALIVE_PLANT = 0
SCORE_ALIVE_MOWER = 0

# Score 10/10
# Scorebis : 10/20