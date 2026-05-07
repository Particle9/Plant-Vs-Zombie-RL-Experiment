import gym
from agents import experienceReplayBuffer, DDQNAgent, QNetwork
import torch
from agents import evaluate
from copy import deepcopy
from pvz import Scene, WaveZombieSpawner, Sunflower, Peashooter, Wallnut, Potatomine, Repeater, Jalapeno, Chomper, config


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
    target_state = target_network.network.state_dict()
    source_state = source_network.network.state_dict() if hasattr(source_network, "network") else source_network.state_dict()

    overlap = min(target_state["0.weight"].shape[1], source_state["0.weight"].shape[1])
    target_state["0.weight"][:, :overlap] = source_state["0.weight"][:, :overlap].clone()
    target_state["0.bias"][: source_state["0.bias"].shape[0]] = source_state["0.bias"].clone()

    output_overlap = min(target_state["2.weight"].shape[0], source_state["2.weight"].shape[0])
    target_state["2.weight"][:output_overlap] = source_state["2.weight"][:output_overlap].clone()
    target_state["2.bias"][:output_overlap] = source_state["2.bias"][:output_overlap].clone()

    source_plants = list(source_env.plant_deck.keys())
    target_plants = list(target_env.plant_deck.keys())
    source_plants_index = {name: i for i, name in enumerate(source_plants)}
    target_plants_index = {name: i for i, name in enumerate(target_plants)}

    for plant_name in source_plants:
        old_plant_index = source_plants_index[plant_name]
        new_plant_index = target_plants_index[plant_name]
        for lane in range(config.N_LANES):
            for pos in range(config.LANE_LENGTH):
                old_action = 1 + old_plant_index + len(source_plants) * (lane + config.N_LANES * pos)
                new_action = 1 + new_plant_index + len(target_plants) * (lane + config.N_LANES * pos)
                target_state["2.weight"][new_action] = source_state["2.weight"][old_action].clone()
                target_state["2.bias"][new_action] = source_state["2.bias"][old_action].clone()

    target_network.network.load_state_dict(target_state)



if __name__ == "__main__":
    # n_iter = 100000
    n_iter = 1000
    env = gym.make('gym_pvz:pvz-env-v2', disable_env_checker=True)
    nn_name = input("Save name: ")
    buffer = experienceReplayBuffer(memory_size=100000, burn_in=10000)
    net = QNetwork(env, device='cpu', use_zombienet=False, use_gridnet=False)
    source_env = make_legacy_ddqn_env()
    source_agent = torch.load(BASE_DDQN_CHECKPOINT, weights_only=False)
    transfer_qnetwork_weights(net, source_agent, source_env, env)
    agent = DDQNAgent(env, net, buffer, n_iter=n_iter, batch_size=200)
    completed = False
    interrupted = False
    try:
        agent.train(max_episodes=n_iter, evaluate_frequency=5000, evaluate_n_iter=1000)
        completed = True
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted. Saving latest checkpoint...")
    finally:
        if completed or interrupted:
            torch.save(agent.network, nn_name)
            agent._save_training_data(nn_name)
            print(f"Saved latest checkpoint to {nn_name}")
