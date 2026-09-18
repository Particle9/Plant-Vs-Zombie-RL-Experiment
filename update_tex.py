import sys

def main():
    with open('1DL073.tex', 'r') as f:
        content = f.read()
        
    old_str = "Action masking"
    new_str = """To address the large action space and the sparsity of valid actions, we implement invalid action masking for all learning agents. At each timestep, the environment provides a boolean mask indicating which of the 316 actions are currently legal. The "wait" action is always valid, while planting actions are only valid if the agent has sufficient sun, the selected plant is not on cooldown, and the target grid cell is empty.

For the DDQN agent, the mask is applied during the greedy action selection phase. The Q-values for all invalid actions are set to a large negative number before taking the argmax, ensuring that an invalid action is never selected as the best action. During the exploration phase, the agent randomly selects only from the pool of valid actions. When computing the target Q-values for the Bellman update, invalid actions in the next state are similarly masked out to prevent the agent from bootstrapping values from impossible future states.

For the Actor-Critic agent, the mask is applied directly to the policy network's output logits. The logits corresponding to invalid actions are masked with a large negative value ($-10^9$) before the softmax activation is applied. This guarantees that the probability of selecting any invalid action is exactly zero, allowing the policy gradient to focus entirely on optimizing the distribution over valid actions."""

    if old_str in content:
        content = content.replace(old_str, new_str)
        with open('1DL073.tex', 'w') as f:
            f.write(content)
        print("Replaced successfully")
    else:
        print("String not found")

if __name__ == "__main__":
    main()
