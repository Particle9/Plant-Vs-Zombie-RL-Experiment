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
# from game_render import render


def _current_scene_stats(env):
    if hasattr(env, "current_chrono"):
        chrono = env.current_chrono()
        base_env = getattr(env, "env", env)
        base_env = getattr(base_env, "unwrapped", base_env)
        scene = getattr(base_env, "_scene", None)
    else:
        base_env = getattr(env, "env", env)
        base_env = getattr(base_env, "unwrapped", base_env)
        scene = getattr(base_env, "_scene", None)
        chrono = getattr(scene, "_chrono", 0) if scene is not None else 0

    wave = 0
    if scene is not None:
        wave = getattr(getattr(scene, "_zombie_spawner", None), "_wave_index", 0)

    return chrono, wave


def evaluate(env, agent, n_iter=1000, verbose = True):
    sum_score = 0
    sum_iter = 0
    sum_wave = 0
    score_hist = []
    iter_hist = []
    wave_hist = []
    n_iter = n_iter
    actions = []

    for episode_idx in range(n_iter):
        if verbose:
            print("\r{}/{}".format(episode_idx, n_iter), end="")
        
        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])

        score_hist.append(summary['score'])
        ticks, wave = _current_scene_stats(env)
        ticks = min(ticks, config.MAX_SECONDS)
        iter_hist.append(ticks)
        # wave index from the scene's spawner (how far the agent reached)
        wave_hist.append(wave)

        sum_score += summary['score']
        sum_iter += ticks
        sum_wave += wave
        
        # if env.env._scene._chrono >= 1000:
        #    render_info = env.env._scene._render_info
        #    render(render_info)
        #    input()
        actions.append(summary['actions'])

    actions = np.concatenate(actions)
    plant_action = np.mod(actions - 1, 4)
    if verbose:
        # Plot of the score
        plt.hist(score_hist)
        plt.title("Score per play over {} plays".format(n_iter))
        plt.show()
        # Plot of the iterations (ticks survived)
        plt.hist(iter_hist)
        plt.title("Survived seconds per play over {} plays".format(n_iter))
        plt.show()
        # Plot of the waves reached
        plt.hist(wave_hist)
        plt.title("Wave index reached per play over {} plays".format(n_iter))
        plt.show()
        # Plot of the action
        plt.hist(np.concatenate(actions), np.arange(0, config.N_LANES * config.LANE_LENGTH * 4 + 2) -0.5, density=True)
        plt.title("Action usage density over {} plays".format(n_iter))
        plt.show()
        plt.hist(plant_action, np.arange(0,5) - 0.5, density=True)
        plt.title("Plant usage density over {} plays".format(n_iter))
        plt.show()

    avg_wave = sum_wave / n_iter if n_iter > 0 else 0
    return sum_score/n_iter, sum_iter/n_iter, avg_wave
