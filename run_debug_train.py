import numpy as np
from agents.actor_critic_agent_v3 import TrainerAC3, ACAgent3

def update_agent_rollouts(agent, summary, rollout_horizon=64):
    num_steps = len(summary['actions'])
    for start in range(0, num_steps, rollout_horizon):
        end = min(start + rollout_horizon, num_steps)
        is_final_chunk = end == num_steps
        next_observation = summary.get('next_observation') if is_final_chunk else summary['observations'][end]
        done = summary.get('done', True) if is_final_chunk else False
        agent.update(
            summary['observations'][start:end],
            summary['actions'][start:end],
            summary['rewards'][start:end],
            summary['dones'][start:end],
            summary['log_probs'][start:end],
            summary['values'][start:end],
            summary['masks'][start:end],
            next_observation=next_observation,
            done=done,
        )

def main():
    trainer = TrainerAC3(render=False, max_waves=25)
    obs_size = trainer.num_observations()
    actions = trainer.get_actions()
    agent = ACAgent3(obs_size, actions)

    n_episodes = 20
    rewards = []
    for ep in range(n_episodes):
        summary = trainer.play(agent)
        total = float(np.sum(summary['rewards']))
        raw_total = float(np.sum(summary['raw_rewards']))
        rewards.append(raw_total)
        print(f"Episode {ep+1}/{n_episodes} training reward: {total}, raw score: {raw_total}")
        update_agent_rollouts(agent, summary)

    print("Raw scores:", rewards)

if __name__ == '__main__':
    main()
