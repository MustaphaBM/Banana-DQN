import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from agent import Agent
from train_agent import train_dqn

if __name__ == "__main__":
    env = UnityEnvironment(
        "/home/mustapha/Desktop/UdacityProjects/banana/Value-based-methods/p1_navigation/Banana_Linux/Banana.x86_64"
    )

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)

    # examine the state space
    state = env_info.visual_observations[0]
    print("States look like:")
    plt.imshow(np.squeeze(state))
    plt.show()
    state_size = state.shape
    print("States have shape:", state.shape)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    scores = train_dqn(
        env=env,
        brain=brain_name,
        agent=agent,
        n_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.show()
