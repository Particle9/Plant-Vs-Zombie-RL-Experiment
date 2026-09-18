import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import os


def save_training_plots(name, show=True):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    figure_dir = os.path.join(repo_root, "agents", "figures", name)
    os.makedirs(figure_dir, exist_ok=True)

    rewards = np.load(name+"_rewards.npy")
    try:
        raw_rewards = np.load(name+"_raw_rewards.npy")
    except Exception:
        raw_rewards = rewards
    iterations = np.load(name+"_iterations.npy")
    try:
        waves = np.load(name+"_waves.npy")
    except Exception:
        waves = None
    loss = torch.load(name+"_loss", weights_only=False)
    real_rewards = np.load(name+"_real_rewards.npy")
    real_iterations = np.load(name+"_real_iterations.npy")
    # optional: real_waves if available
    try:
        real_waves = np.load(name+"_real_waves.npy")
    except Exception:
        real_waves = None

    slice_size = 500

    def downsample(values, window):
        values = np.asarray(values)
        if len(values) == 0:
            return np.array([])
        if len(values) <= window:
            return values
        chunk_count = len(values) // window
        trimmed = values[:chunk_count * window]
        if trimmed.size == 0:
            return values
        return np.reshape(trimmed, (chunk_count, window)).mean(axis=1)

    rewards = downsample(rewards, slice_size)
    raw_rewards = downsample(raw_rewards, slice_size)
    iterations = downsample(iterations, slice_size)
    if waves is not None:
        waves = downsample(waves, slice_size)
    loss = downsample(loss, slice_size)

    x = np.arange(1, len(rewards) + 1) * slice_size
    real_x = np.arange(1, len(real_rewards) + 1) * slice_size
    plt.figure()
    plt.plot(x, rewards)
    if len(real_rewards) > 0:
        plt.plot(real_x, real_rewards, color='red')
    plt.title("Shaped reward over training")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(figure_dir, "score.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.figure()
    plt.plot(x, raw_rewards, color='purple')
    if len(real_rewards) > 0:
        plt.plot(real_x, real_rewards, color='red')
    plt.title("Raw score over training")
    plt.xlabel("Iteration")
    plt.ylabel("Raw score")
    plt.savefig(os.path.join(figure_dir, "raw_score.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.figure()
    plt.plot(x, iterations)
    if len(real_iterations) > 0:
        plt.plot(real_x, real_iterations, color='red')
    plt.title("Ticks survived over training")
    plt.xlabel("Iteration")
    plt.ylabel("Ticks")
    plt.savefig(os.path.join(figure_dir, "ticks_survived.png"), bbox_inches="tight")
    if show:
        plt.show()
    if real_waves is not None:
        plt.figure()
        if len(real_waves) > 0:
            plt.plot(real_x, real_waves, color='green')
        plt.title('Wave survived over training')
        plt.xlabel('Iteration')
        plt.ylabel('Wave')
        plt.savefig(os.path.join(figure_dir, "waves_survived.png"), bbox_inches="tight")
        if show:
            plt.show()
    if waves is not None:
        plt.figure()
        plt.plot(x, waves, color='teal')
        if real_waves is not None and len(real_waves) > 0:
            plt.plot(real_x, real_waves, color='green')
        plt.title('Training wave over training')
        plt.xlabel('Iteration')
        plt.ylabel('Wave')
        plt.savefig(os.path.join(figure_dir, "training_waves.png"), bbox_inches="tight")
        if show:
            plt.show()


if __name__ == "__main__":
    save_training_plots(sys.argv[1], show=True)
    # plt.plot(x, loss)
    # plt.show()
