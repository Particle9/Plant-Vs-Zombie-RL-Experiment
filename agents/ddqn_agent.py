import torch.nn as nn
import torch
import torch.nn.functional as F
import gym
from pvz import config
from copy import deepcopy
from collections import deque
import numpy as np
import gc
import resource
from .threshold import Threshold
from . import evaluate

HP_NORM = 4000.0
SUN_NORM = 500.0
LANE_HP_NORM = 6000.0
MAX_COOLDOWN_NORM = 50.0
MAX_TRAINING_HISTORY = 5000
MEMORY_LOG_EPISODES = 100

def _num_observations(plant_deck):
    return 2 * config.N_LANES * config.LANE_LENGTH + (2 * len(plant_deck)) + (4 * config.N_LANES) + 4


def _danger_score(scene):
    danger = 0.0
    for zombie in scene.zombies:
        distance_weight = (config.LANE_LENGTH - zombie.pos) / max(1, config.LANE_LENGTH)
        danger += zombie.hp * (0.5 + distance_weight)
    return danger


def _disable_render_history(env):
    scene = getattr(env.unwrapped, "_scene", None)
    if scene is not None:
        scene._record_render_history = False
        scene._render_info = []


def _scene_snapshot(scene):
    return {
        "zombie_hp": sum(zombie.hp for zombie in scene.zombies),
        "danger": _danger_score(scene),
        "wave": getattr(scene._zombie_spawner, "_wave_index", 0),
        "mowers": int(scene.grid._mowers.astype(np.int32).sum()),
        "lives": int(scene.lives),
    }


def _memory_usage_mb():
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB.
    if rss_kb > 10**7:
        return rss_kb / (1024 * 1024)
    return rss_kb / 1024


def _transform_observation(observation, scene, plant_deck, max_waves):
    grid_size = config.N_LANES * config.LANE_LENGTH
    observation = observation.astype(np.float32)
    plant_grid = observation[:grid_size] / max(1.0, float(len(plant_deck)))
    zombie_grid = np.clip(observation[grid_size:(2 * grid_size)] / HP_NORM, 0.0, 4.0)
    sun = np.array([min(observation[2 * grid_size] / SUN_NORM, 4.0)], dtype=np.float32)
    action_available = observation[2 * grid_size + 1:].astype(np.float32)
    cooldowns = np.array(
        [scene.plant_cooldowns[plant_name] / MAX_COOLDOWN_NORM for plant_name in plant_deck],
        dtype=np.float32,
    )
    mowers = scene.grid._mowers.astype(np.float32)
    lane_zombie_hp = np.zeros(config.N_LANES, dtype=np.float32)
    lane_plant_hp = np.zeros(config.N_LANES, dtype=np.float32)
    nearest_zombie = np.ones(config.N_LANES, dtype=np.float32)

    for zombie in scene.zombies:
        lane_zombie_hp[zombie.lane] += zombie.hp
        nearest_zombie[zombie.lane] = min(
            nearest_zombie[zombie.lane],
            zombie.pos / max(1, config.LANE_LENGTH - 1),
        )

    for plant in scene.plants:
        lane_plant_hp[plant.lane] += plant.hp

    lane_zombie_hp = np.clip(lane_zombie_hp / LANE_HP_NORM, 0.0, 4.0)
    lane_plant_hp = np.clip(lane_plant_hp / LANE_HP_NORM, 0.0, 4.0)
    wave_index = getattr(scene._zombie_spawner, "_wave_index", 0)
    wave_progress = min(wave_index / max(1, config.MAX_WAVE), 1.0)
    wave_limit_progress = min(wave_index / max(1, max_waves), 1.5)
    chrono_progress = min(scene._chrono / max(1.0, config.MAX_SECONDS), 1.0)
    global_features = np.array([wave_progress, wave_limit_progress, chrono_progress], dtype=np.float32)

    return np.concatenate([
        plant_grid,
        zombie_grid,
        sun,
        action_available,
        cooldowns,
        mowers,
        lane_zombie_hp,
        lane_plant_hp,
        nearest_zombie,
        global_features,
    ])


def _shape_reward(prev_stats, current_stats, reward):
    prev_zombie_hp = prev_stats["zombie_hp"]
    current_zombie_hp = current_stats["zombie_hp"]
    prev_danger = prev_stats["danger"]
    current_danger = current_stats["danger"]
    prev_wave = prev_stats["wave"]
    current_wave = current_stats["wave"]
    prev_mowers = prev_stats["mowers"]
    current_mowers = current_stats["mowers"]

    score_reward = reward / 25.0
    damage_reward = max(prev_zombie_hp - current_zombie_hp, 0.0) / 40.0
    pressure_reward = (prev_danger - current_danger) / 120.0
    wave_delta = max(current_wave - prev_wave, 0)
    wave_reward = (20.0 + 3.0 * current_wave) * wave_delta
    lane_clear_bonus = 8.0 if current_danger < prev_danger and current_wave >= 10 else 0.0
    life_penalty = -350.0 * max(prev_stats["lives"] - current_stats["lives"], 0)
    mower_penalty = -100.0 * max(prev_mowers - current_mowers, 0)
    shaped_reward = score_reward + damage_reward + pressure_reward + wave_reward + lane_clear_bonus + life_penalty + mower_penalty
    return float(np.clip(shaped_reward, -250.0, 250.0))


class QNetwork(nn.Module):
    
    def __init__(self, env, epsilon=0.05, learning_rate=1e-3, device='cpu', use_zombienet=True, use_gridnet=True):
        super(QNetwork, self).__init__()
        self.device = device

        self.n_inputs = _num_observations(env.plant_deck)
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        # Set up network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 256, bias=True),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, self.n_outputs, bias=True))

        # Set to GPU if cuda is specified
        if self.device == 'cuda':
            self.network.cuda()
            
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=self.learning_rate)
        
    def decide_action(self, state, mask, epsilon):
        # mask = self.env.mask_available_actions()
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions[mask])
        else:
            action = self.get_greedy_action(state, mask)
        return action
    
    def get_greedy_action(self, state, mask):
        with torch.inference_mode():
            qvals = self.get_qvals(state)
            qvals[np.logical_not(mask)] = qvals.min()
            return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state, use_zombienet=True):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to(device=self.device)
        else:
            state_t = torch.FloatTensor(state).to(device=self.device)

        # Keep compatibility with legacy checkpoints whose first layer expects
        # a different observation size than the current environment produces.
        expected_inputs = self.network[0].in_features
        if state_t.dim() == 1:
            current_inputs = state_t.shape[0]
            if current_inputs > expected_inputs:
                state_t = state_t[:expected_inputs]
            elif current_inputs < expected_inputs:
                pad = torch.zeros(expected_inputs - current_inputs, device=state_t.device, dtype=state_t.dtype)
                state_t = torch.cat((state_t, pad), dim=0)
        elif state_t.dim() == 2:
            current_inputs = state_t.shape[1]
            if current_inputs > expected_inputs:
                state_t = state_t[:, :expected_inputs]
            elif current_inputs < expected_inputs:
                pad = torch.zeros(
                    (state_t.shape[0], expected_inputs - current_inputs),
                    device=state_t.device,
                    dtype=state_t.dtype,
                )
                state_t = torch.cat((state_t, pad), dim=1)
        return self.network(state_t)


class ZombieNet(nn.Module):
    def __init__(self, output_size=1, hidden_size=5):
        super(ZombieNet, self).__init__()
        self.fc1 = nn.Linear(config.LANE_LENGTH, output_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x = F.leaky_relu(self.fc1(x))
        # x = self.fc2(x)
        return self.fc1(x)

class DDQNAgent:
    
    def __init__(self, env, network, buffer, n_iter = 100000, batch_size=32, max_waves=25):
        
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        self.max_waves = max_waves
        # self.pre_buffer = []
        # self.pre_buffer_rewards = []
        # self.threshold = Threshold(seq_length = 100000, start_epsilon=1.0,
        #                   end_epsilon=0.2,interpolation='sinusoidal',
        #                   periods=np.floor(n_iter/100))
        self.threshold = Threshold(seq_length = n_iter, start_epsilon=1.0, interpolation="exponential",
                           end_epsilon=0.05)
        self.epsilon = 0
        self.batch_size = batch_size
        self.window = 100
        self.tau = 1.0
        # Keep training running until max_episodes or interruption unless an explicit threshold is set.
        self.reward_threshold = None
        self.initialize()
        self.player = PlayerQ(render=False, max_waves=max_waves)
        _disable_render_history(self.env)
        _disable_render_history(self.player.env)
        

    def take_step(self, mode='train'):
        mask = np.array(self.env.mask_available_actions())
        if mode == 'explore':
            if np.random.random()<0.5:
                action=0 # Do nothing
            else:
                action = np.random.choice(np.arange(self.env.action_space.n)[mask])
        else:
            action = self.network.decide_action(self.s_0, mask, epsilon=self.epsilon)
            self.step_count += 1
        prev_stats = _scene_snapshot(self.env.unwrapped._scene)
        s_1, r, done, _ = self.env.step(action)
        current_wave = getattr(self.env.unwrapped._scene._zombie_spawner, "_wave_index", 0)
        reached_wave_limit = current_wave > self.max_waves
        done = done or reached_wave_limit
        current_stats = _scene_snapshot(self.env.unwrapped._scene)
        shaped_reward = _shape_reward(prev_stats, current_stats, r)
        s_1 = self._transform_observation(s_1)
        self.rewards += shaped_reward
        self.raw_rewards += r
        self.buffer.append(self.s_0, action, shaped_reward, done, s_1)
        self.s_0 = s_1.copy()
        if done:
            if mode != "explore": # We document the end of the play
                self.training_iterations.append(self.env.unwrapped._scene._chrono)
                self.training_waves.append(min(current_wave, self.max_waves))
                self.training_raw_rewards.append(self.raw_rewards)
            self.s_0 = self._transform_observation(self.env.reset())
            _disable_render_history(self.env)
        return done
    
    # def add_play_to_buffer(self):
    #     rewards = self.discount_rewards(np.array(self.pre_buffer_rewards))
    #     for i in range(len(rewards)):
    #         s_0, action, done, s_1 = self.pre_buffer[i]
    #         r = rewards[i]
    #         self.buffer.append(s_0, action, r, done, s_1)
    #     self.pre_buffer_rewards = []
    #     self.pre_buffer = []
        
    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=100000,
              network_update_frequency=32,
              network_sync_frequency=200,
              evaluate_frequency=500,
              evaluate_n_iter=1000,
              curriculum_fraction=0.5):

        self.gamma = gamma
        base_max_waves = self.max_waves
        curriculum_start_waves = min(6, base_max_waves)
        curriculum_span = max(1, int(max_episodes * curriculum_fraction))
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')
            # if done:
            #     self.add_play_to_buffer()
        ep = 0
        training = True
        self.s_0 = self._transform_observation(self.env.reset())
        _disable_render_history(self.env)


        while training:
            if curriculum_fraction > 0:
                progress = min(1.0, ep / curriculum_span)
                self.max_waves = int(curriculum_start_waves + (base_max_waves - curriculum_start_waves) * progress)
            self.rewards = 0
            self.raw_rewards = 0
            done = False
            while done == False:
                self.epsilon = self.threshold.epsilon(ep)
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self._soft_update_target_network()
                    self.sync_eps.append(ep)
                    
                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(tuple(self.training_rewards)[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    mean_raw_rewards = np.mean(tuple(self.training_raw_rewards)[-self.window:])
                    self.mean_training_raw_rewards.append(mean_raw_rewards)

                    mean_iteration = np.mean(tuple(self.training_iterations)[-self.window:])
                    self.mean_training_iterations.append(mean_iteration)
                    mean_wave = np.mean(tuple(self.training_waves)[-self.window:])
                    self.mean_training_waves.append(mean_wave)
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t Mean Raw {:.2f}\t\t Mean Iterations {:.2f}\t\t Mean Wave {:.2f}\t\t".format(
                        ep, mean_rewards, mean_raw_rewards, mean_iteration, mean_wave), end="")
                    if ep % MEMORY_LOG_EPISODES == 0:
                        print(
                            "\n[ddqn memory] episode={} rss_mb={:.1f} replay_size={} updates_pending={}".format(
                                ep,
                                _memory_usage_mb(),
                                self.buffer.current_size,
                                len(self.update_loss),
                            )
                        )
                    
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if self.reward_threshold is not None and mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        break
                    if (ep % evaluate_frequency) == evaluate_frequency - 1:
                        self.player.max_waves = base_max_waves
                        avg_score, avg_iter, avg_wave = evaluate(self.player, self.network, n_iter=evaluate_n_iter, verbose=False)
                        self.real_iterations.append(avg_iter)
                        self.real_rewards.append(avg_score)
                        self.real_waves.append(avg_wave)
                        gc.collect()
                    elif ep % MEMORY_LOG_EPISODES == 0:
                        gc.collect()


                    

    def calculate_loss(self, batch):
        full_mask = np.full(self.env.action_space.n, True)

        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1,1)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1,1).to(
            device=self.network.device)
        dones_t = torch.BoolTensor(dones).to(device=self.network.device)

        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t) # The selected action already respects the mask
        
        #################################################################
        # DDQN Update
        next_masks = np.array([self._get_mask(s) for s in next_states])
        with torch.no_grad():
            qvals_next_pred = self.network.get_qvals(next_states)
            qvals_next_pred[np.logical_not(next_masks)] = -1e9
            next_actions = torch.max(qvals_next_pred, dim=-1)[1]
            next_actions_t = torch.LongTensor(next_actions).reshape(-1,1).to(
                device=self.network.device)
            target_qvals = self.target_network.get_qvals(next_states)
            qvals_next = torch.gather(target_qvals, 1, next_actions_t)
        #################################################################
        qvals_next[dones_t] = 0 # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = F.smooth_l1_loss(qvals, expected_qvals)
        return loss
    
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def _transform_observation(self, observation):
        return _transform_observation(
            observation,
            self.env.unwrapped._scene,
            self.env.plant_deck,
            self.max_waves,
        )

    def _get_mask(self, observation):
        empty_cells = np.nonzero((observation[:self._grid_size] == 0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(self.env.action_space.n, dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * len(self.env.plant_deck)

        availability_start = (2 * self._grid_size) + 1
        availability_end = availability_start + len(self.env.plant_deck)
        available_plants = observation[availability_start:availability_end]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM
        
    def _save_training_data(self, nn_name):
        np.save(nn_name+"_rewards", np.asarray(self.training_rewards, dtype=np.float32))
        np.save(nn_name+"_raw_rewards", np.asarray(self.training_raw_rewards, dtype=np.float32))
        np.save(nn_name+"_iterations", np.asarray(self.training_iterations, dtype=np.float32))
        np.save(nn_name+"_waves", np.asarray(self.training_waves, dtype=np.float32))
        np.save(nn_name+"_real_rewards", np.asarray(self.real_rewards, dtype=np.float32))
        np.save(nn_name+"_real_iterations", np.asarray(self.real_iterations, dtype=np.float32))
        np.save(nn_name+"_real_waves", np.asarray(self.real_waves, dtype=np.float32))
        torch.save(list(self.training_loss), nn_name+"_loss")

    def _soft_update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def initialize(self):
        self.training_rewards = deque(maxlen=MAX_TRAINING_HISTORY)
        self.training_raw_rewards = deque(maxlen=MAX_TRAINING_HISTORY)
        self.training_loss = deque(maxlen=MAX_TRAINING_HISTORY)
        self.training_iterations = deque(maxlen=MAX_TRAINING_HISTORY)
        self.training_waves = deque(maxlen=MAX_TRAINING_HISTORY)
        self.real_rewards = deque(maxlen=MAX_TRAINING_HISTORY)
        self.real_iterations = deque(maxlen=MAX_TRAINING_HISTORY)
        self.real_waves = deque(maxlen=MAX_TRAINING_HISTORY)
        self.update_loss = []
        self.mean_training_rewards = deque(maxlen=MAX_TRAINING_HISTORY)
        self.mean_training_raw_rewards = deque(maxlen=MAX_TRAINING_HISTORY)
        self.mean_training_iterations = deque(maxlen=MAX_TRAINING_HISTORY)
        self.mean_training_waves = deque(maxlen=MAX_TRAINING_HISTORY)
        self.sync_eps = deque(maxlen=MAX_TRAINING_HISTORY)
        self.rewards = 0
        self.raw_rewards = 0
        self.step_count = 0
        self.s_0 = self._transform_observation(self.env.reset())
        _disable_render_history(self.env)

class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.states = None
        self.next_states = None
        self.actions = np.empty(memory_size, dtype=np.int32)
        self.rewards = np.empty(memory_size, dtype=np.float32)
        self.dones = np.empty(memory_size, dtype=np.bool_)
        self.write_index = 0
        self.current_size = 0

    def _ensure_storage(self, state):
        if self.states is not None:
            return
        state_arr = np.asarray(state, dtype=np.float32)
        state_shape = state_arr.shape
        self.states = np.empty((self.memory_size,) + state_shape, dtype=np.float32)
        self.next_states = np.empty((self.memory_size,) + state_shape, dtype=np.float32)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(self.current_size, batch_size, replace=False)
        return (
            self.states[samples],
            self.actions[samples],
            self.rewards[samples],
            self.dones[samples],
            self.next_states[samples],
        )

    def append(self, state, action, reward, done, next_state):
        self._ensure_storage(state)
        self.states[self.write_index] = np.asarray(state, dtype=np.float32)
        self.actions[self.write_index] = action
        self.rewards[self.write_index] = reward
        self.dones[self.write_index] = done
        self.next_states[self.write_index] = np.asarray(next_state, dtype=np.float32)
        self.write_index = (self.write_index + 1) % self.memory_size
        self.current_size = min(self.current_size + 1, self.memory_size)

    def burn_in_capacity(self):
        return self.current_size / self.burn_in


class PlayerQ():
    def __init__(self, env = None, render=True, max_waves=25):
        if env==None:
            self.env = gym.make('gym_pvz:pvz-env-v2', disable_env_checker=True)
        else:
            self.env = env
        self.render = render
        self.max_waves = max_waves
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        _disable_render_history(self.env)

        
    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return _num_observations(self.env.plant_deck)

    def num_actions(self):
        return self.env.action_space.n

    def _transform_observation(self, observation):
        return _transform_observation(
            observation,
            self.env.unwrapped._scene,
            self.env.plant_deck,
            self.max_waves,
        )

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM

    def play(self,agent, epsilon=0):
        """ Play one episode and collect observations and rewards """

        summary = dict()
        summary['rewards'] = list()
        summary['raw_rewards'] = list()
        summary['observations'] = list()
        summary['actions'] = list()
        observation = self._transform_observation(self.env.reset())
        _disable_render_history(self.env)
        
        t = 0

        while getattr(self.env.unwrapped._scene._zombie_spawner, "_wave_index", 0) <= self.max_waves:
            if(self.render):
                self.env.render()
            # if np.random.random()<epsilon:
            #     # print("exploration")
            #     action = np.random.choice(self.get_actions(), 1)[0]
            # else:
            # action = agent.decide_action(observation, np.full(self.num_actions(), True), epsilon)
            action = agent.decide_action(observation, self.env.mask_available_actions(), epsilon)
            summary['observations'].append(observation)
            summary['actions'].append(action)
            observation, reward, done, info = self.env.step(action)
            observation = self._transform_observation(observation)
            summary['rewards'].append(reward)
            summary['raw_rewards'].append(reward)

            if done or getattr(self.env.unwrapped._scene._zombie_spawner, "_wave_index", 0) > self.max_waves:
                break

        summary['observations'] = np.asarray(summary['observations'], dtype=np.float32)
        summary['actions'] = np.asarray(summary['actions'], dtype=np.int64)
        summary['raw_rewards'] = np.asarray(summary['raw_rewards'], dtype=np.float32)
        summary['rewards'] = np.asarray(summary['rewards'], dtype=np.float32)
        _disable_render_history(self.env)
        return summary

    def get_render_info(self):
        return self.env.unwrapped._scene._render_info
