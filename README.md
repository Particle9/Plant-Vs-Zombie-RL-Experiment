# Plants vs. Zombies: Reinforcement Learning Experiment

**Course:** Natural Computation For Machine Learning (1DL073)

This project is an advanced reinforcement learning and heuristic AI experiment built upon the `pvz_rl` Open AI Gym environment. We extend the original Plants vs. Zombies RL framework to implement, debug, and compare multiple artificial intelligence approaches, including Deep Q-Networks (DDQN), Actor-Critic methods, and custom rule-based heuristics.

## ✨ Features & Enhancements

This repository significantly extends the base environment with the following additions:

- **Heuristic Agent Baseline**: A robust, rule-based AI prioritizing emergency defense, economy generation, and offensive pressure. It serves as a highly interpretable and strong baseline for evaluating RL models (see [HEURISTIC_AGENT.md](HEURISTIC_AGENT.md)).
- **Actor-Critic Agent**: A fully implemented and debugged Actor-Critic RL agent featuring fixed gradient flows, value function detached gradients, and proper action masking (see [CONVERGENCE_FIXES.md](CONVERGENCE_FIXES.md)).
- **Extended DDQN Training**: Scripts to fine-tune and transfer-learn DDQN models on the complete plant deck (including Chomper, Repeater, and Jalapeno).
- **Agent Comparison Framework**: Comprehensive Jupyter notebooks (`comparison.ipynb`) to benchmark multiple agents across hundreds of episodes, generating summary statistics and visualizations.

## ⚙️ Installation & Setup

The game engine and the Open AI gym environment are encapsulated in Python libraries. To run this project locally, you will need Python 3 installed along with PyTorch, Gym, PyGame, and optionally SHAP.

Clone the repository and install the local packages:

```bash
git clone <your-repo-url>
cd Plant-Vs-Zombie-RL-Experiment

# Install the base PvZ engine
cd pvz
pip install -e .
cd ..

# Install the Gym environment
cd gym-pvz
pip install -e .
cd ..
```

To verify your installation, run the render script to watch the trained DDQN agent play:
```bash
python game_render.py
```

## 🚀 Usage

### 1. Training Agents

You can train various RL agents from scratch using the dedicated training scripts. 
*Note: Training can take upwards of 30+ minutes depending on your hardware.*

**Train DDQN Agent:**
```bash
python train_ddqn_agent.py
```
*This script uses transfer learning from an existing checkpoint (`agents/agent_zoo/dfq5_epsexp`) to fine-tune a model on the full plant deck.*

**Train Actor-Critic Agent:**
```bash
python train_actor_critic_agent.py
```

### 2. Evaluating and Comparing Agents

**Jupyter Notebook Comparison:**
The easiest way to compare the agents (DDQN, Actor-Critic, Heuristic, Random) is to run the provided notebook:
- Open `comparison.ipynb`
- Run the cells to benchmark agents across 100+ episodes and visualize the results (score distributions, survival ticks, and wave progression).

**Script-based Evaluation:**
To evaluate a specific trained agent, modify `script_evaluate.py` to point to your saved model checkpoint, then run:
```bash
python script_evaluate.py
```

### 3. Rendering a Game

To visually observe an agent's behavior:
```bash
python game_render.py
```
By default, this renders the DDQN agent. You can point `DDQN_RENDER_MODEL` in the script to your newly trained checkpoint to evaluate custom agents. To increase the simulation speed, adjust the FPS variable in `pvz/pvz/config.py`.

### 4. Feature Importance (SHAP)

For DDQN agents, you can analyze which features the network prioritizes using SHAP values:
```bash
python script_feature_importance.py
```

## 📁 Repository Structure

- `/agents`: Contains agent implementations (`heuristic_agent.py`, actor critic, etc.) and the agent zoo (checkpoints).
- `/pvz` & `/gym-pvz`: The core game engine and OpenAI Gym wrapper.
- `/entities`: Plant and zombie characteristics and balancing logic.
- `comparison.ipynb`: Multi-agent benchmarking notebook.
- `train_*.py`: Training scripts for various RL architectures.

---
*Based on the original [pvz_rl project](https://github.com/inf581-pvz-anonymous/pvz_rl)*
