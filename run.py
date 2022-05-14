import matplotlib.pyplot as plt
import numpy as np
import yaml
from unityagents import UnityEnvironment

from agent import Agent
from train_agent import train_dqn

if __name__ == "__main__":
    env = UnityEnvironment(
        "/home/mustapha/Desktop/udacity_nano_degree/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64"
    )

    with open("config.yaml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    agent_hyperparams = params["hyperparameters"]
    training_params = params["trainingparameters"]
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print("States have shape:", state.shape)

    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        seed=0,
        agent_hyperparams=agent_hyperparams,
    )

    scores = train_dqn(env=env, brain=brain_name, agent=agent, **training_params)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.show()
