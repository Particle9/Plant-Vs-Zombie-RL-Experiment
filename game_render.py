from agents import ReinforceAgentV2, PolicyNetV2, PlayerV2
from agents import KeyboardAgent
from agents import PlayerQ, QNetwork, QNetwork_DQN
from agents import ACAgent3, TrainerAC3
from pvz import config
import os
import gym
import torch
import pygame

PLANT_RENDER_ORDER = [
    "sunflower",
    "peashooter",
    "wall-nut",
    "potatomine",
    "chomper",
    "repeater",
    "jalapeno",
]

class PVZ():
    def __init__(self,render=True, max_frames = 1000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        
    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * (config.LANE_LENGTH + 2)

    def num_actions(self):
        return self.env.action_space.n

    def play(self,agent):
        """ Play one episode and collect observations and rewards """
        observation = self.env.reset()
        t = 0

        for t in range(self.max_frames):
            if(self.render):
                self.env.render()

            action = agent.decide_action(observation)
            observation, reward, done, info = self.env.step(action)

            if done:
                break

    def get_render_info(self):
        return self.env._scene._render_info

def render(render_info):
    pygame.init()
    pygame.font.init() # you have to call this at the start, 
                    # if you want to use this module.
    myfont = pygame.font.SysFont('calibri', 30)


    screen = pygame.display.set_mode((1450, 650))
    zombie_sprite = {"zombie": pygame.image.load("assets/zombie_scaled.png").convert_alpha(),
     "zombie_cone": pygame.image.load("assets/zombie_cone_scaled.png").convert_alpha(),
     "zombie_bucket": pygame.image.load("assets/zombie_bucket_scaled.png").convert_alpha(),
        "zombie_flag" :pygame.image.load("assets/zombie_flag_scaled.png").convert_alpha(),
        "zombie_newspaper" :pygame.image.load("assets/zombie_newspaper_scaled.png").convert_alpha(),
        "zombie_pole_vault" :pygame.image.load("assets/zombie_pole_vault_scaled.png").convert_alpha(),
        "zombie_all_star" :pygame.image.load("assets/zombie_allstar_scaled.png").convert_alpha(),  }
    plant_sprite = {"peashooter": pygame.image.load("assets/peashooter_scaled.png").convert_alpha(),
                    "sunflower": pygame.image.load("assets/sunflower_scaled.png").convert_alpha(),
                    "wallnut": pygame.image.load("assets/wallnut_scaled.png").convert_alpha(),
                    "potatomine":pygame.image.load("assets/potatomine_scaled.png").convert_alpha(),
                    "chomper":pygame.image.load("assets/chomper_scaled.png").convert_alpha(),
                    "repeater":pygame.image.load("assets/repeater_scaled.png").convert_alpha(),
                    "jalapeno":pygame.image.load("assets/jalapeno_scaled.png").convert_alpha()}
    projectile_sprite = {
        "pea": pygame.image.load("assets/pea.png").convert_alpha(),
        "mower": pygame.image.load("assets/mower_scaled.png").convert_alpha(),
    }
    clock = pygame.time.Clock()
    cell_size = 75
    offset_border = 100
    offset_y = int(0.8 * cell_size)
    cumulated_score=0

    while render_info:
        clock.tick(config.FPS)
        screen.fill((130, 200, 100))
        frame_info = render_info.pop(0)
        
        # The grid
        for i in range(config.LANE_LENGTH+1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border + i * cell_size, offset_border), 
                (offset_border + i * cell_size, offset_border + cell_size * (config.N_LANES)), 1)
        for j in range(config.N_LANES+1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border, offset_border + j * cell_size), 
                (offset_border + cell_size * (config.LANE_LENGTH), offset_border + j * cell_size), 1)
        
        
        # The objects
        for lane in range(config.N_LANES):
            for zombie_name, pos, offset in frame_info["zombies"][lane]:
                zombie_name = zombie_name.lower()
                screen.blit(zombie_sprite[zombie_name], (offset_border + cell_size * (pos + offset) - zombie_sprite[zombie_name].get_width(),
                    offset_border + lane * cell_size + offset_y - zombie_sprite[zombie_name].get_height()))
            if frame_info.get("mowers", [False] * config.N_LANES)[lane]:
                sprite = projectile_sprite["mower"]
                screen.blit(sprite, (offset_border - sprite.get_width(),
                    offset_border + lane * cell_size + offset_y - sprite.get_height()))
            for plant_name, pos in frame_info["plants"][lane]:
                plant_name = plant_name.lower()
                screen.blit(plant_sprite[plant_name], (offset_border + cell_size * pos, 
                    offset_border + lane * cell_size + offset_y - plant_sprite[plant_name].get_height()))
            for projectile_name, pos, offset in frame_info["projectiles"][lane]:
                projectile_name = projectile_name.lower()
                sprite = projectile_sprite.get(projectile_name)
                if sprite is None:
                    continue
                screen.blit(sprite, (offset_border + cell_size * (pos+offset) - sprite.get_width(), 
                    offset_border + lane * cell_size))
        
        #Text
        sun_text = myfont.render('Sun: '+ str(frame_info["sun"]), False, (0, 0, 0))
        screen.blit(sun_text, (50, 600))
        cumulated_score += frame_info["score"]
        score_text = myfont.render('Score: '+ str(cumulated_score), False, (0, 0, 0))
        screen.blit(score_text, (200, 600))
        wave_text = myfont.render('Wave: '+ str(frame_info["wave"]), False, (0, 0, 0))
        screen.blit(wave_text, (450, 600))
        budget_text = myfont.render('Budget: '+ str(frame_info["budget"]), False, (0, 0, 0))
        screen.blit(budget_text, (650, 600))
        time = myfont.render('Time: '+ str(frame_info["time"]), False, (0, 0, 0))
        screen.blit(time, (900, 100))
        cooldowns = frame_info["cooldowns"]
        cooldown_lines = ['Cooldowns:']
        cooldown_lines.extend([
            plant_name.replace('-', ' ').title() + ': ' + str(cooldowns.get(plant_name, 0))
            for plant_name in PLANT_RENDER_ORDER
        ])
        for i, line in enumerate(cooldown_lines):
            cooldown_text = myfont.render(line, False, (0, 0, 0))
            screen.blit(cooldown_text, (900, 135 + i * 25))
        
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                render_info = []
        
        
        
        pygame.display.flip()

    pygame.quit()

agent_type = "DDQN" # DDQN or Reinforce or AC or Keyboard
DDQN_RENDER_MODEL = os.environ.get("PVZ_DDQN_RENDER_MODEL", "agents/agent_zoo/ddqn_new")


if __name__ == "__main__":

    if agent_type == "Reinforce":
        env = PlayerV2(render=False, max_frames = 500 * config.FPS)
        agent = ReinforceAgentV2(
                input_size=env.num_observations(),
                possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/dfp5")
        
    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        agent = torch.load(DDQN_RENDER_MODEL, weights_only=False)
        
    if agent_type == "AC":
        env = TrainerAC3(render=False, max_frames = 500*config.FPS)
        agent = ACAgent3(
                input_size=env.num_observations(),
                possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/ac_policy_v1", "agents/agent_zoo/ac_value_v1")
    
    if agent_type == "Keyboard":
        env = PlayerV2(render=True, max_frames = 500*config.FPS)
        agent = KeyboardAgent()
    env.play(agent)
    render_info = env.get_render_info()
    render(render_info)
