import tiles3 as tc
import numpy as np
from agent import BaseAgent
import mountaincar_env
from rl_glue1 import RLGlue
import time
from matplotlib import pyplot as plt
from tqdm import tqdm


def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)

class MountainCarTileCoder:
    def __init__(self,iht_size=4096,num_tilings=8,num_tiles=8):
        self.ith = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self,position,velocity):
        POSITION_MIN = -1.2
        POSITION_MAX = 0.5
        VELOCITY_MIN = -0.07
        VELOCITY_MAX = 0.07

        position_scale = self.num_tiles/(POSITION_MAX-POSITION_MIN)
        velocity_scale = self.num_tiles/(VELOCITY_MAX-VELOCITY_MIN)

        return tc.tiles(self.ith,self.num_tilings,[position*position_scale,velocity*velocity_scale])

class SarsaAgent(BaseAgent):
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.epsilon = None
        self.gamma = None
        self.iht_size = None
        self.w = None
        self.alpha = None
        self.num_tilings = None
        self.num_tiles = None
        self.mctc = None
        self.initial_weights = None
        self.num_actions = None
        self.previous_tiles = None

    def agent_init(self, agent_info= {}):
        self.num_tilings = agent_info.get("num_tilings", 8)
        self.num_tiles = agent_info.get("num_tiles", 8)
        self.iht_size = agent_info.get("iht_size", 4096)
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.gamma = agent_info.get("gamma", 1.0)
        self.alpha = agent_info.get("alpha", 0.5) / self.num_tilings
        self.initial_weights = agent_info.get("initial_weights", 0.0)
        self.num_actions = agent_info.get("num_actions", 3)
        self.w = np.ones((self.num_actions,self.iht_size))*self.initial_weights
        self.tc = MountainCarTileCoder(iht_size=self.iht_size,num_tilings=self.num_tilings,num_tiles=self.num_tiles)

    def select_action(self,tiles):
        action_values = []
        chosen_action = None
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            action_values[action] = np.sum(self.w[action][tiles])
        if np.random.random()< self.epsilon:
            chosen_action = np.random.randint(0,self.num_actions)
        else:
            chosen_action = np.argmax(action_values)

        return chosen_action,action_values[chosen_action]

    def agent_start(self, state):
        position,velocity = state
        active_tiles = self.tc.get_tiles(position,velocity)
        current_action,action_value = self.select_action(active_tiles)

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action


    def agent_step(self, reward, state):
        position,velocity = state

        active_tiles = self.tc.get_tiles(position,velocity)
        current_action,action_value = self.select_action(active_tiles)
        last_action_value = np.sum(self.w[self.last_action][self.previous_tiles])
        self.w[self.last_action][self.previous_tiles] += self.alpha *(reward+self.gamma*action_value-last_action_value)
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_end(self, reward):

        last_action_value = np.sum(self.w[self.last_action][self.previous_tiles])
        self.w[self.last_action][self.previous_tiles] += self.alpha*(reward-last_action_value)

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass


# Compare the three
num_runs = 1
num_episodes = 100
env_info = {}

agent_runs = []
# alphas = [0.2, 0.4, 0.5, 1.0]
alphas = [0.5]
agent_info_options = [{"num_tiles": 16, "num_tilings": 2, "alpha": 0.5},
                      {"num_tiles": 4, "num_tilings": 32, "alpha": 0.5},
                      {"num_tiles": 8, "num_tilings": 8, "alpha": 0.5}]
agent_info_options = [{"num_tiles": agent["num_tiles"],
                       "num_tilings": agent["num_tilings"],
                       "alpha": alpha} for agent in agent_info_options for alpha in alphas]

agent = SarsaAgent
env = mountaincar_env.Environment
for agent_info in agent_info_options:
    all_steps = []
    start = time.time()
    for run in tqdm(range(num_runs)):
        if run % 5 == 0:
            print("RUN: {}".format(run))
        env = mountaincar_env.Environment

        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        steps_per_episode = []

        for episode in range(num_episodes):
            rl_glue.rl_episode(15000)
            steps_per_episode.append(rl_glue.num_steps)
        all_steps.append(np.array(steps_per_episode))

    agent_runs.append(np.mean(np.array(all_steps), axis=0))
    print(rl_glue.agent.alpha)
    print("Run Time: {}".format(time.time() - start))

plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.array(agent_runs).T)
plt.xlabel("Episode")
plt.ylabel("Steps Per Episode")
plt.yscale("linear")
plt.ylim(0, 1000)
plt.legend(["num_tiles: {}, num_tilings: {}, alpha: {}".format(agent_info["num_tiles"],
                                                               agent_info["num_tilings"],
                                                               agent_info["alpha"])
            for agent_info in agent_info_options])
plt.show()