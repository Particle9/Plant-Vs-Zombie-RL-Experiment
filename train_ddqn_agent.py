import gym
from agents import experienceReplayBuffer, DDQNAgent, QNetwork
import torch
from agents import evaluate
from copy import deepcopy
from pvz import Scene, WaveZombieSpawner, Sunflower, Peashooter, Wallnut, Potatomine, Repeater, Jalapeno, Chomper, config
from plot_training_ddqn import save_training_plots


BASE_DDQN_CHECKPOINT = "agents/agent_zoo/dfq5_epsexp"


def make_legacy_ddqn_env():
    env = gym.make('gym_pvz:pvz-env-v2', disable_env_checker=True).unwrapped
    env.plant_deck = {
        "sunflower": Sunflower,
        "peashooter": Peashooter,
        "wall-nut": Wallnut,
        "potatomine": Potatomine,
    }
    env.action_space = gym.spaces.Discrete(len(env.plant_deck) * config.N_LANES * config.LANE_LENGTH + 1)
    env.observation_space = gym.spaces.Tuple([
        gym.spaces.MultiDiscrete([len(env.plant_deck) + 1] * (config.N_LANES * config.LANE_LENGTH)),
        gym.spaces.MultiDiscrete([10000] * (config.N_LANES * config.LANE_LENGTH)),
        gym.spaces.Discrete(10000),
        gym.spaces.MultiBinary(len(env.plant_deck)),
    ])
    env._plant_names = [plant_name for plant_name in env.plant_deck]
    env._plant_classes = [env.plant_deck[plant_name].__name__ for plant_name in env.plant_deck]
    env._plant_no = {env._plant_classes[i]: i for i in range(len(env._plant_names))}
    env._scene = Scene(env.plant_deck, WaveZombieSpawner())
    return env


def transfer_qnetwork_weights(target_network, source_network, source_env, target_env):
    """Transfer only middle layer weights from source to target network.
    
    The middle layers contain learned feature representations that transfer well.
    First layer and output layer are left uninitialized to adapt to the different
    input/output dimensions between source and target networks.
    """
    target_state = target_network.network.state_dict()
    source_state = source_network.network.state_dict() if hasattr(source_network, "network") else source_network.state_dict()
    
    # Get all weight/bias layer keys sorted by order
    target_weight_keys = sorted([key for key in target_state.keys() if key.endswith("weight")], 
                               key=lambda x: int(x.split('.')[0]))
    source_weight_keys = sorted([key for key in source_state.keys() if key.endswith("weight")], 
                               key=lambda x: int(x.split('.')[0]))
    target_bias_keys = sorted([key for key in target_state.keys() if key.endswith("bias")], 
                             key=lambda x: int(x.split('.')[0]))
    source_bias_keys = sorted([key for key in source_state.keys() if key.endswith("bias")], 
                             key=lambda x: int(x.split('.')[0]))
    
    # Transfer only middle layers (skip first and last)
    # First layer: adapts to input encoding differences
    # Last layer: adapts to action space differences
    middle_range = range(1, len(target_weight_keys) - 1)
    
    for i in middle_range:
        if i < len(source_weight_keys) - 1:
            target_key = target_weight_keys[i]
            source_key = source_weight_keys[i]
            target_bias_key = target_bias_keys[i]
            source_bias_key = source_bias_keys[i]
            
            # Transfer weights with dimension matching
            target_shape = target_state[target_key].shape
            source_shape = source_state[source_key].shape
            
            if target_shape == source_shape:
                # Perfect fit, copy directly
                target_state[target_key] = source_state[source_key].clone()
                target_state[target_bias_key] = source_state[source_bias_key].clone()
            else:
                # Partial transfer: copy overlapping dimensions
                min_out = min(target_shape[0], source_shape[0])
                min_in = min(target_shape[1], source_shape[1])
                target_state[target_key][:min_out, :min_in] = source_state[source_key][:min_out, :min_in].clone()
                target_state[target_bias_key][:min_out] = source_state[source_bias_key][:min_out].clone()
    
    target_network.network.load_state_dict(target_state)



if __name__ == "__main__":
    n_iter = 100000
    # n_iter = 5000
    max_waves = 100
    env = gym.make('gym_pvz:pvz-env-v2', disable_env_checker=True)
    nn_name = "agents/agent_zoo/ddqn_tfx"
    buffer = experienceReplayBuffer(memory_size=10000, burn_in=5000)
    net = QNetwork(env, learning_rate=1e-3, device='cpu', use_zombienet=True, use_gridnet=False)
    source_env = make_legacy_ddqn_env()
    source_agent = torch.load(BASE_DDQN_CHECKPOINT, weights_only=False)
    transfer_qnetwork_weights(net, source_agent, source_env, env)
    agent = DDQNAgent(env, net, buffer, n_iter=n_iter, batch_size=128, max_waves=max_waves)
    completed = False
    interrupted = False
    try:
        agent.train(max_episodes=n_iter, network_update_frequency=16, network_sync_frequency=128, evaluate_frequency=100, evaluate_n_iter=25)
        completed = True
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted. Saving latest checkpoint...")
    finally:
        if completed or interrupted:
            torch.save(agent.network, nn_name)
            agent._save_training_data(nn_name)
            save_training_plots(nn_name, show=False)
            print(f"Saved latest checkpoint to {nn_name}")
