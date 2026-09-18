import gym
from itertools import count
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pvz import config
import matplotlib.pyplot as plt
from torch.distributions import Categorical

HP_NORM = 4000.0
SUN_NORM = 500.0
LANE_HP_NORM = 6000.0
MAX_COOLDOWN_NORM = 50.0

class PolicynetAC3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=192):
        super(PolicynetAC3, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.affine1(x)))
        x = F.leaky_relu(self.norm2(self.affine2(x)))
        action_logits = self.action_head(x)
        return action_logits

class ValuenetAC3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=192):
        super(ValuenetAC3, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.affine1(x)))
        x = F.leaky_relu(self.norm2(self.affine2(x)))
        state_value = self.value_head(x)
        return state_value


class ACAgent3():
    def __init__(self,input_size, possible_actions):
        self.possible_actions = possible_actions
        self.policy = PolicynetAC3(input_size, output_size=len(possible_actions))
        self.valuenet = ValuenetAC3(input_size, output_size=len(possible_actions))
        self.optimizer1 = optim.Adam(self.policy.parameters(), lr=2.5e-4)
        self.optimizer2 = optim.Adam(self.valuenet.parameters(), lr=2.5e-4)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 1.5e-2
        self.value_coef = 0.5
        self.reward_scale = 1.0
        self.clip_eps = 0.2
        self.ppo_epochs = 5
        self.batch_size = 128
        self.max_grad_norm = 0.5

    def _distribution(self, states, masks=None):
        logits = self.policy(states)
        if masks is not None:
            logits = logits.masked_fill(~masks, -1e9)
        return Categorical(logits=logits)

    def decide_action(self, state, mask=None, deterministic=False):
        state_tensor = torch.from_numpy(state).float()
        mask_tensor = None if mask is None else torch.from_numpy(mask).bool()
        with torch.no_grad():
            dist = self._distribution(state_tensor, mask_tensor)
            value = self.valuenet(state_tensor).squeeze(-1)
            action = torch.argmax(dist.logits) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), float(log_prob.item()), float(value.item())


    def update(self, observation, actions, rewards, dones, old_log_probs, values, masks, next_observation=None, done=True):
        if len(observation) == 0:
            return

        observations = torch.tensor(observation, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).view(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1) / self.reward_scale
        dones = torch.tensor(dones, dtype=torch.float32).view(-1)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).view(-1)
        old_values = torch.tensor(values, dtype=torch.float32).view(-1)
        masks = torch.tensor(masks, dtype=torch.bool)

        if next_observation is not None and not done:
            next_state = torch.tensor(next_observation, dtype=torch.float32)
            with torch.no_grad():
                next_value = self.valuenet(next_state).view(())
        else:
            next_value = torch.tensor(0.0)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = torch.tensor(0.0)

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - float(done)
            else:
                next_val = old_values[step + 1]
                next_non_terminal = 1.0 - dones[step]

            delta = rewards[step] + self.gamma * next_val * next_non_terminal - old_values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + old_values[step]

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_steps = observations.shape[0]
        indices = np.arange(num_steps)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_steps, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                obs_batch = observations[batch_idx]
                action_batch = actions[batch_idx]
                old_log_prob_batch = old_log_probs[batch_idx]
                advantage_batch = advantages[batch_idx]
                return_batch = returns[batch_idx]
                mask_batch = masks[batch_idx]

                dist = self._distribution(obs_batch, mask_batch)
                new_log_probs = dist.log_prob(action_batch)
                entropy_bonus = dist.entropy().mean()
                new_values = self.valuenet(obs_batch).squeeze(-1)

                ratio = torch.exp(new_log_probs - old_log_prob_batch)
                unclipped = ratio * advantage_batch
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage_batch
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.smooth_l1_loss(new_values, return_batch)

                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.valuenet.parameters(), self.max_grad_norm)
                self.optimizer1.step()
                self.optimizer2.step()

    def save(self, nn_name_1, nn_name_2):
        torch.save(self.policy, nn_name_1)
        torch.save(self.valuenet, nn_name_2)


    def load(self, nn_name_1, nn_name_2):
        # These checkpoints were saved as full module objects, so disable
        # weights_only for PyTorch >=2.6 compatibility.
        self.policy = torch.load(nn_name_1, weights_only=False)
        self.valuenet = torch.load(nn_name_2, weights_only=False)




class TrainerAC3():
    def __init__(self,render=True, max_waves=20, n_iter = 100000):
        self.env = gym.make('gym_pvz:pvz-env-v2', disable_env_checker=True)
        self.max_waves = max_waves
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self._disable_render_history()

    def _disable_render_history(self):
        scene = getattr(self.env.unwrapped, "_scene", None)
        if scene is not None:
            scene._record_render_history = False
            scene._render_info = []

    def _scene(self):
        return self.env.unwrapped._scene


    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        base_obs = 2 * config.N_LANES * config.LANE_LENGTH + len(self.env.plant_deck) + 1
        extra_obs = len(self.env.plant_deck) + (4 * config.N_LANES) + 2
        return base_obs + extra_obs

    def num_actions(self):
        return self.env.action_space.n

    def _transform_observation(self, observation):
        scene = self._scene()
        observation = observation.astype(np.float32)
        plant_grid = observation[:self._grid_size] / max(1.0, float(len(self.env.plant_deck)))
        zombie_grid = np.clip(observation[self._grid_size:(2 * self._grid_size)] / HP_NORM, 0.0, 4.0)
        sun = np.array([min(observation[2 * self._grid_size] / SUN_NORM, 4.0)], dtype=np.float32)
        action_available = observation[2 * self._grid_size + 1:].astype(np.float32)

        cooldowns = np.array(
            [scene.plant_cooldowns[plant_name] / MAX_COOLDOWN_NORM for plant_name in self.env.plant_deck],
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
        wave_progress = min(getattr(scene._zombie_spawner, "_wave_index", 0) / max(1, config.MAX_WAVE), 1.0)
        wave_limit_progress = min(
            getattr(scene._zombie_spawner, "_wave_index", 0) / max(1.0, self.max_waves),
            1.5,
        )
        global_features = np.array([wave_progress, wave_limit_progress], dtype=np.float32)

        observation = np.concatenate([
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
        if self.render:
            print(observation)
        return observation

    def _danger_score(self, scene):
        danger = 0.0
        for zombie in scene.zombies:
            distance_weight = (config.LANE_LENGTH - zombie.pos) / max(1, config.LANE_LENGTH)
            danger += zombie.hp * (0.5 + distance_weight)
        return danger

    def play(self,agent, deterministic=False):
        """ Play one episode and collect observations and rewards """

        summary = dict()
        summary['rewards'] = list()
        summary['raw_rewards'] = list()
        summary['observations'] = list()
        summary['actions'] = list()
        summary['dones'] = list()
        summary['log_probs'] = list()
        summary['values'] = list()
        summary['masks'] = list()
        observation = self._transform_observation(self.env.reset())
        self._disable_render_history()
        done = False

        while getattr(self._scene()._zombie_spawner, "_wave_index", 0) <= self.max_waves:
            if(self.render):
                self.env.render()

            # Get action mask for valid actions
            mask = self.env.unwrapped.mask_available_actions()
            scene_before = self._scene()
            prev_zombie_hp = sum(zombie.hp for zombie in scene_before.zombies)
            prev_lives = scene_before.lives
            prev_wave = getattr(scene_before._zombie_spawner, "_wave_index", 0)
            prev_danger = self._danger_score(scene_before)
            prev_mowers = scene_before.grid._mowers.astype(np.int32).sum()
            action, log_prob, value = agent.decide_action(observation, mask, deterministic=deterministic)

            summary['observations'].append(observation)
            summary['actions'].append(action)
            summary['log_probs'].append(log_prob)
            summary['values'].append(value)
            summary['masks'].append(mask.astype(np.bool_))
            observation, reward, done, info = self.env.step(action)
            observation = self._transform_observation(observation)
            scene_after = self._scene()
            current_zombie_hp = sum(zombie.hp for zombie in scene_after.zombies)
            current_danger = self._danger_score(scene_after)
            current_wave = getattr(scene_after._zombie_spawner, "_wave_index", 0)
            current_mowers = scene_after.grid._mowers.astype(np.int32).sum()

            score_reward = reward / 25.0
            damage_reward = max(prev_zombie_hp - current_zombie_hp, 0.0) / 40.0
            pressure_reward = (prev_danger - current_danger) / 120.0
            wave_delta = max(current_wave - prev_wave, 0)
            wave_reward = (20.0 + 3.0 * current_wave) * wave_delta
            lane_clear_bonus = 8.0 if current_danger < prev_danger and current_wave >= 10 else 0.0
            life_penalty = -350.0 * max(prev_lives - scene_after.lives, 0)
            mower_penalty = -100.0 * max(prev_mowers - current_mowers, 0)
            shaped_reward = score_reward + damage_reward + pressure_reward + wave_reward + lane_clear_bonus + life_penalty + mower_penalty
            shaped_reward = float(np.clip(shaped_reward, -250.0, 250.0))

            summary['raw_rewards'].append(reward)
            summary['rewards'].append(shaped_reward)
            summary['dones'].append(done)

            if done:
                break

        summary['observations'] = np.asarray(summary['observations'], dtype=np.float32)
        summary['actions'] = np.asarray(summary['actions'], dtype=np.int64)
        summary['rewards'] = np.asarray(summary['rewards'], dtype=np.float32)
        summary['raw_rewards'] = np.asarray(summary['raw_rewards'], dtype=np.float32)
        summary['dones'] = np.asarray(summary['dones'], dtype=np.bool_)
        summary['log_probs'] = np.asarray(summary['log_probs'], dtype=np.float32)
        summary['values'] = np.asarray(summary['values'], dtype=np.float32)
        summary['masks'] = np.asarray(summary['masks'], dtype=np.bool_)
        summary['next_observation'] = observation
        summary['done'] = done
        self._disable_render_history()
        return summary

    def get_render_info(self):
        return self._scene()._render_info

    def current_chrono(self):
        return self._scene()._chrono
