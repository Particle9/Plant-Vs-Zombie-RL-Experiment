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
import os
import numpy as np
from collections import deque

MAX_TRAINING_HISTORY = 5000


def _current_scene_stats(env):
    """Extract current chrono (ticks) and wave index from the environment."""
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


def _update_agent_rollouts(agent, summary, rollout_horizon=64):
    num_steps = len(summary["actions"])
    for start in range(0, num_steps, rollout_horizon):
        end = min(start + rollout_horizon, num_steps)
        is_final_chunk = end == num_steps
        next_observation = summary.get("next_observation") if is_final_chunk else summary["observations"][end]
        done = summary.get("done", True) if is_final_chunk else False
        agent.update(
            summary["observations"][start:end],
            summary["actions"][start:end],
            summary["rewards"][start:end],
            summary["dones"][start:end],
            summary["log_probs"][start:end],
            summary["values"][start:end],
            summary["masks"][start:end],
            next_observation=next_observation,
            done=done,
        )


def _run_greedy_evaluation(env, agent, eval_max_waves):
    original_max_waves = env.max_waves
    env.max_waves = eval_max_waves
    try:
        summary = env.play(agent, deterministic=True)
        score = float(np.sum(summary["rewards"]))
        raw_score = float(np.sum(summary["raw_rewards"]))
        ticks, wave = _current_scene_stats(env)
        reached_wave = min(wave, eval_max_waves)
        return score, raw_score, ticks, reached_wave
    finally:
        env.max_waves = original_max_waves


def train(env, agent, n_iter=100000, n_record=500, n_save=1000, n_eval=100, n_greedy_eval=100, model_name="ac_model", rollout_horizon=96, curriculum_fraction=0.5):
    sum_score = 0
    sum_raw_score = 0
    sum_iter = 0
    sum_wave = 0
    score_plt = []
    raw_score_plt = []
    iter_plt = []
    wave_plt = []
    eval_score_plt = []
    eval_iter_plt = []
    save = False
    best_score = None
    # Keep rolling metrics bounded so very long runs do not keep growing in RAM.
    history_limit = max(1, min(n_iter, MAX_TRAINING_HISTORY))
    greedy_history_limit = max(1, min(MAX_TRAINING_HISTORY, (n_iter // max(1, n_greedy_eval)) + 1))
    real_rewards = deque(maxlen=history_limit)
    real_iterations = deque(maxlen=history_limit)
    real_waves = deque(maxlen=history_limit)
    eval_episodes = deque(maxlen=history_limit)
    greedy_eval_rewards = deque(maxlen=greedy_history_limit)
    greedy_eval_raw_scores = deque(maxlen=greedy_history_limit)
    greedy_eval_iterations = deque(maxlen=greedy_history_limit)
    greedy_eval_waves = deque(maxlen=greedy_history_limit)
    greedy_eval_episodes = deque(maxlen=greedy_history_limit)
    base_max_waves = env.max_waves
    curriculum_start_waves = min(6, base_max_waves)
    curriculum_span = max(1, int(n_iter * curriculum_fraction))

    for episode_idx in range(n_iter):
        if curriculum_fraction > 0:
            progress = min(1.0, episode_idx / curriculum_span)
            env.max_waves = int(curriculum_start_waves + (base_max_waves - curriculum_start_waves) * progress)

        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])
        summary['raw_score'] = np.sum(summary["raw_rewards"])

        sum_score += summary['score']
        sum_raw_score += summary['raw_score']
        sum_iter += env.current_chrono()
        _, wave_stats = _current_scene_stats(env)
        sum_wave += wave_stats

        # Update agent
        _update_agent_rollouts(agent, summary, rollout_horizon=rollout_horizon)

        if n_greedy_eval > 0 and ((episode_idx + 1) % n_greedy_eval == 0):
            greedy_score, greedy_raw_score, greedy_iter, greedy_wave = _run_greedy_evaluation(
                env,
                agent,
                eval_max_waves=base_max_waves,
            )
            greedy_eval_rewards.append(greedy_score)
            greedy_eval_raw_scores.append(greedy_raw_score)
            greedy_eval_iterations.append(greedy_iter)
            greedy_eval_waves.append(greedy_wave)
            greedy_eval_episodes.append(episode_idx + 1)

        if (episode_idx%n_record == n_record-1):
            if save:
                if sum_score >= best_score:
                    agent.save(nn_name1, nn_name2)
                    best_score = sum_score
            print("---Episode {}, mean training reward {}".format(episode_idx,sum_score/n_record))
            print("---Episode {}, mean raw score {}".format(episode_idx,sum_raw_score/n_record))
            print("---Episode {}, mean wave {}".format(episode_idx,sum_wave/n_record))
            print("---n_iter {}".format(sum_iter/n_record))
            score_plt.append(sum_score/n_record)
            raw_score_plt.append(sum_raw_score/n_record)
            iter_plt.append(sum_iter/n_record)
            wave_plt.append(sum_wave/n_record)
            sum_iter = 0
            sum_score = 0
            sum_raw_score = 0
            sum_wave = 0
            # input()
            # save interim arrays and plots per n_record (plots show points averaged per n_eval)
            repo_root = os.path.dirname(os.path.abspath(__file__))
            figure_dir = os.path.join(repo_root, "agents", "figures", model_name)
            os.makedirs(figure_dir, exist_ok=True)
            # compute per-n_eval averages from evaluation arrays
            rewards_arr = np.asarray(real_rewards, dtype=np.float32)
            iters_arr = np.asarray(real_iterations, dtype=np.float32)
            waves_arr = np.asarray(real_waves, dtype=np.float32)
            num_points = len(rewards_arr) // n_eval
            if num_points > 0:
                batch_x = np.arange(n_eval, n_eval * num_points + 1, n_eval)
                y_rewards = [np.mean(rewards_arr[i * n_eval:(i + 1) * n_eval]) for i in range(num_points)]
                y_iters = [np.mean(iters_arr[i * n_eval:(i + 1) * n_eval]) for i in range(num_points)]
                y_waves = [np.mean(waves_arr[i * n_eval:(i + 1) * n_eval]) for i in range(num_points)]
                plt.figure()
                plt.plot(batch_x, y_rewards, color='blue')
                plt.title("Training reward over batches")
                plt.xlabel("Training episode")
                plt.ylabel("Reward")
                plt.savefig(os.path.join(figure_dir, "reward_batch.png"), bbox_inches="tight")
                plt.close()
                if raw_score_plt:
                    plt.figure()
                    plt.plot(np.arange(n_record, n_record * len(raw_score_plt) + 1, n_record), raw_score_plt, color='purple')
                    plt.title("Raw score over batches")
                    plt.xlabel("Training episode")
                    plt.ylabel("Raw score")
                    plt.savefig(os.path.join(figure_dir, "raw_score_batch.png"), bbox_inches="tight")
                    plt.close()
                plt.figure()
                plt.plot(batch_x, y_iters, color='red')
                plt.title("Ticks survived over training batches")
                plt.xlabel("Training episode")
                plt.ylabel("Ticks")
                plt.savefig(os.path.join(figure_dir, "ticks_batch.png"), bbox_inches="tight")
                plt.close()
                plt.figure()
                plt.plot(batch_x, y_waves, color='green')
                plt.title("Waves survived over training batches")
                plt.xlabel("Training episode")
                plt.ylabel("Wave")
                plt.savefig(os.path.join(figure_dir, "waves_batch.png"), bbox_inches="tight")
                plt.close()
            np.save(os.path.join(repo_root, model_name+"_real_rewards"), np.asarray(real_rewards, dtype=np.float32))
            np.save(os.path.join(repo_root, model_name+"_real_iterations"), np.asarray(real_iterations, dtype=np.float32))
            np.save(os.path.join(repo_root, model_name+"_real_waves"), np.asarray(real_waves, dtype=np.float32))
            np.save(os.path.join(repo_root, model_name+"_greedy_eval_rewards"), np.asarray(greedy_eval_rewards, dtype=np.float32))
            np.save(os.path.join(repo_root, model_name+"_greedy_eval_raw_scores"), np.asarray(greedy_eval_raw_scores, dtype=np.float32))
            np.save(os.path.join(repo_root, model_name+"_greedy_eval_iterations"), np.asarray(greedy_eval_iterations, dtype=np.float32))
            np.save(os.path.join(repo_root, model_name+"_greedy_eval_waves"), np.asarray(greedy_eval_waves, dtype=np.float32))
            np.save(os.path.join(repo_root, model_name+"_greedy_eval_episodes"), np.asarray(greedy_eval_episodes, dtype=np.int32))
            try:
                x = np.asarray(eval_episodes, dtype=np.int32)
                plt.figure()
                plt.plot(x, np.asarray(real_iterations, dtype=np.float32), color='red')
                plt.title("Ticks survived (evaluation)")
                plt.xlabel("Training episode")
                plt.ylabel("Ticks")
                plt.savefig(os.path.join(figure_dir, "ticks_survived_eval.png"), bbox_inches="tight")
                plt.close()
                plt.figure()
                plt.plot(x, np.asarray(real_waves, dtype=np.float32), color='green')
                plt.title("Waves survived (evaluation)")
                plt.xlabel("Training episode")
                plt.ylabel("Wave")
                plt.savefig(os.path.join(figure_dir, "waves_survived_eval.png"), bbox_inches="tight")
                plt.close()
                plt.figure()
                plt.plot(x, np.asarray(real_rewards, dtype=np.float32), color='blue')
                plt.title("Reward (evaluation)")
                plt.xlabel("Training episode")
                plt.ylabel("Reward")
                plt.savefig(os.path.join(figure_dir, "reward_eval.png"), bbox_inches="tight")
                plt.close()
                if greedy_eval_episodes:
                    greedy_x = np.asarray(greedy_eval_episodes, dtype=np.int32)
                    plt.figure()
                    plt.plot(greedy_x, np.asarray(greedy_eval_raw_scores, dtype=np.float32), color='purple')
                    plt.title("Greedy raw score")
                    plt.xlabel("Training episode")
                    plt.ylabel("Raw score")
                    plt.savefig(os.path.join(figure_dir, "greedy_raw_score_eval.png"), bbox_inches="tight")
                    plt.close()
                    plt.figure()
                    plt.plot(greedy_x, np.asarray(greedy_eval_iterations, dtype=np.float32), color='red')
                    plt.title("Greedy ticks survived")
                    plt.xlabel("Training episode")
                    plt.ylabel("Ticks")
                    plt.savefig(os.path.join(figure_dir, "greedy_ticks_eval.png"), bbox_inches="tight")
                    plt.close()
                    plt.figure()
                    plt.plot(greedy_x, np.asarray(greedy_eval_waves, dtype=np.float32), color='green')
                    plt.title("Greedy waves survived")
                    plt.xlabel("Training episode")
                    plt.ylabel("Wave")
                    plt.savefig(os.path.join(figure_dir, "greedy_waves_eval.png"), bbox_inches="tight")
                    plt.close()
            except Exception:
                pass
        
        # evaluation each episode - extract metrics from current training episode
        avg_score = summary['score']
        avg_iter, avg_wave = _current_scene_stats(env)
        real_rewards.append(avg_score)
        real_iterations.append(avg_iter)
        real_waves.append(avg_wave)
        eval_episodes.append(episode_idx + 1)
        
        if not save:
            if (episode_idx%n_save == n_save-1):
                save = True
                best_score = 0
                nn_name1 = model_name + "_policy.pt"
                nn_name2 = model_name + "_value.pt"
                # s = input("Save? (y/n): ")
                # if (s=='y'):
                #     save = True
                #     best_score = 0
                #     nn_name1 = input("Save name for policy net: ")
                #     nn_name2 = input("Save name for value net: ")

    plt.figure(200)
    plt.plot(range(n_record, n_iter+1, n_record), score_plt)
    plt.show()
    plt.figure(300)
    plt.plot(range(n_record, n_iter+1, n_record), iter_plt)
    plt.show()
    plt.figure(400)
    plt.plot(range(n_record, n_iter+1, n_record), wave_plt)
    plt.show()
    # save final evaluation arrays and plots if evaluation was used
    if len(real_iterations) > 0 or len(real_waves) > 0:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        figure_dir = os.path.join(repo_root, "agents", "figures", model_name)
        os.makedirs(figure_dir, exist_ok=True)
        np.save(os.path.join(repo_root, model_name+"_real_rewards"), np.asarray(real_rewards, dtype=np.float32))
        np.save(os.path.join(repo_root, model_name+"_real_iterations"), np.asarray(real_iterations, dtype=np.float32))
        np.save(os.path.join(repo_root, model_name+"_real_waves"), np.asarray(real_waves, dtype=np.float32))
        np.save(os.path.join(repo_root, model_name+"_eval_episodes"), np.asarray(eval_episodes, dtype=np.int32))
        np.save(os.path.join(repo_root, model_name+"_greedy_eval_rewards"), np.asarray(greedy_eval_rewards, dtype=np.float32))
        np.save(os.path.join(repo_root, model_name+"_greedy_eval_raw_scores"), np.asarray(greedy_eval_raw_scores, dtype=np.float32))
        np.save(os.path.join(repo_root, model_name+"_greedy_eval_iterations"), np.asarray(greedy_eval_iterations, dtype=np.float32))
        np.save(os.path.join(repo_root, model_name+"_greedy_eval_waves"), np.asarray(greedy_eval_waves, dtype=np.float32))
        np.save(os.path.join(repo_root, model_name+"_greedy_eval_episodes"), np.asarray(greedy_eval_episodes, dtype=np.int32))
        # compute per-n_eval averages for saved batch plots
        rewards_arr = np.asarray(real_rewards, dtype=np.float32)
        iters_arr = np.asarray(real_iterations, dtype=np.float32)
        waves_arr = np.asarray(real_waves, dtype=np.float32)
        num_points = len(rewards_arr) // n_eval
        if num_points > 0:
            batch_x = np.arange(n_eval, n_eval * num_points + 1, n_eval)
            y_rewards = [np.mean(rewards_arr[i * n_eval:(i + 1) * n_eval]) for i in range(num_points)]
            y_iters = [np.mean(iters_arr[i * n_eval:(i + 1) * n_eval]) for i in range(num_points)]
            y_waves = [np.mean(waves_arr[i * n_eval:(i + 1) * n_eval]) for i in range(num_points)]
            plt.figure()
            plt.plot(batch_x, y_rewards, color='blue')
            plt.title("Training reward over batches")
            plt.xlabel("Training episode")
            plt.ylabel("Reward")
            plt.savefig(os.path.join(figure_dir, "reward_batch.png"), bbox_inches="tight")
            plt.close()
            if raw_score_plt:
                plt.figure()
                plt.plot(np.arange(n_record, n_record * len(raw_score_plt) + 1, n_record), raw_score_plt, color='purple')
                plt.title("Raw score over batches")
                plt.xlabel("Training episode")
                plt.ylabel("Raw score")
                plt.savefig(os.path.join(figure_dir, "raw_score_batch.png"), bbox_inches="tight")
                plt.close()
            plt.figure()
            plt.plot(batch_x, y_iters, color='red')
            plt.title("Ticks survived over training batches")
            plt.xlabel("Training episode")
            plt.ylabel("Ticks")
            plt.savefig(os.path.join(figure_dir, "ticks_batch.png"), bbox_inches="tight")
            plt.close()
            plt.figure()
            plt.plot(batch_x, y_waves, color='green')
            plt.title("Waves survived over training batches")
            plt.xlabel("Training episode")
            plt.ylabel("Wave")
            plt.savefig(os.path.join(figure_dir, "waves_batch.png"), bbox_inches="tight")
            plt.close()
        try:
            x = np.asarray(eval_episodes, dtype=np.int32)
            plt.figure()
            plt.plot(x, np.asarray(real_iterations, dtype=np.float32), color='red')
            plt.title("Ticks survived (evaluation)")
            plt.xlabel("Training episode")
            plt.ylabel("Ticks")
            plt.savefig(os.path.join(figure_dir, "ticks_survived_eval.png"), bbox_inches="tight")
            plt.close()
            plt.figure()
            plt.plot(x, np.asarray(real_waves, dtype=np.float32), color='green')
            plt.title("Waves survived (evaluation)")
            plt.xlabel("Training episode")
            plt.ylabel("Wave")
            plt.savefig(os.path.join(figure_dir, "waves_survived_eval.png"), bbox_inches="tight")
            plt.close()
            plt.figure()
            plt.plot(x, np.asarray(real_rewards, dtype=np.float32), color='blue')
            plt.title("Reward (evaluation)")
            plt.xlabel("Training episode")
            plt.ylabel("Reward")
            plt.savefig(os.path.join(figure_dir, "reward_eval.png"), bbox_inches="tight")
            plt.close()
            if greedy_eval_episodes:
                greedy_x = np.asarray(greedy_eval_episodes, dtype=np.int32)
                plt.figure()
                plt.plot(greedy_x, np.asarray(greedy_eval_raw_scores, dtype=np.float32), color='purple')
                plt.title("Greedy raw score")
                plt.xlabel("Training episode")
                plt.ylabel("Raw score")
                plt.savefig(os.path.join(figure_dir, "greedy_raw_score_eval.png"), bbox_inches="tight")
                plt.close()
                plt.figure()
                plt.plot(greedy_x, np.asarray(greedy_eval_iterations, dtype=np.float32), color='red')
                plt.title("Greedy ticks survived")
                plt.xlabel("Training episode")
                plt.ylabel("Ticks")
                plt.savefig(os.path.join(figure_dir, "greedy_ticks_eval.png"), bbox_inches="tight")
                plt.close()
                plt.figure()
                plt.plot(greedy_x, np.asarray(greedy_eval_waves, dtype=np.float32), color='green')
                plt.title("Greedy waves survived")
                plt.xlabel("Training episode")
                plt.ylabel("Wave")
                plt.savefig(os.path.join(figure_dir, "greedy_waves_eval.png"), bbox_inches="tight")
                plt.close()
        except Exception:
            pass


# Import your agent
from agents import ACAgent3, TrainerAC3

if __name__ == "__main__":

    env = TrainerAC3(render=False, max_waves=100)
    agent = ACAgent3(
        input_size = env.num_observations(),
        possible_actions=env.get_actions()
    )
    train(env, agent, n_greedy_eval=100, model_name="ac_t_v6")




