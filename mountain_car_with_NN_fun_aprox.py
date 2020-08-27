import numpy as np
#import mountaincar_env

from mountain_car import MountainCarEnvironment
from rl_glue1 import RLGlue
from matplotlib import pyplot as plt
from tqdm import tqdm
from agent import BaseAgent
from copy import deepcopy
import os
import shutil
import time
from plot_script2 import plot_result


class ActionValueNetwork:
    def __init__(self, networkconfig):
        self.state_dim = networkconfig.get("state_dim")
        self.num_hidden_units = networkconfig.get("num_hidden_units")
        self.num_actions = networkconfig.get("num_actions")
        self.rand_generator = np.random.RandomState(networkconfig.get("seed"))

        self.layer_sizes = np.array([self.state_dim, self.num_hidden_units, self.num_actions])

        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))

    def get_action_values(self, s):

        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)

        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = np.dot(x, W1) + b1

        return q_vals

    def get_TD_update(self, s, delta_mat):
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']

        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)

        td_update = [dict() for i in range(len(self.weights))]

        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        return td_update

    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor

    def get_weights(self):
        """
        Returns:
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)

    # Work Required: No.
    def set_weights(self, weights):
        """
        Args:
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)


class Adam():
    # Work Required: Yes. Fill in the initialization for self.m and self.v (~4 Lines).
    def __init__(self, layer_sizes,
                 optimizer_info):
        self.layer_sizes = layer_sizes

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]

        for i in range(0, len(self.layer_sizes) - 1):
            ### START CODE HERE (~4 Lines)
            # Hint: The initialization for m and v should look very much like the initializations of the weights
            # except for the fact that initialization here is to zeroes (see description above.)
            self.m[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            ### END CODE HERE

        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to
        # the time step t. We can calculate these powers using an incremental product. At initialization then,
        # beta_m_product and beta_v_product should be ...? (Note that timesteps start at 1 and if we were to
        # start from 0, the denominator would be 0.)
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    # Work Required: Yes. Fill in the weight updates (~5-7 lines).
    def update_weights(self, weights, td_errors_times_gradients):
        """
        Args:
            weights (Array of dictionaries): The weights of the neural network.
            td_errors_times_gradients (Array of dictionaries): The gradient of the
            action-values with respect to the network's weights times the TD-error
        Returns:
            The updated weights (Array of dictionaries).
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                ### START CODE HERE (~5-7 Lines)
                # Hint: Follow the equations above. First, you should update m and v and then compute
                # m_hat and v_hat. Finally, compute how much the weights should be incremented by.
                # self.m[i][param] = None
                # self.v[i][param] = None
                # m_hat = None
                # v_hat = None
                ### update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * td_errors_times_gradients[i][
                    param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * (
                            td_errors_times_gradients[i][param] ** 2)
                ### compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                ### update weights
                weight_update = self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
                ### END CODE HERE

                weights[i][param] = weights[i][param] + weight_update
        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to
        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights

class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
        if len(self.buffer)== self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size= self.minibatch_size)
        return[self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)


def softmax(action_values, tau=1.0):

    preferences = action_values / tau
    max_preference = np.max(preferences, axis=1)

    reshaped_max_preference = max_preference.reshape((-1, 1))

    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)

    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    action_probs = action_probs.squeeze()
    return action_probs

def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
    q_next_mat = current_q.get_action_values(next_states)
    probs_mat = softmax(q_next_mat, tau)

    v_next_vec = np.sum(probs_mat * q_next_mat, axis= 1)* (1-terminals)

    target_vec = rewards + discount* v_next_vec

    q_mat = network.get_action_values(states)

    batch_indices = np.arange(q_mat.shape[0])

    q_vec = q_mat[batch_indices, actions]

    delta_vec = target_vec - q_vec

    return delta_vec

def optimize_network(experiences, discount, optimizer, network, current_q, tau):

    states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    batch_size = states.shape[0]

    delta_vec = get_td_error(states,next_states,actions,rewards,discount,terminals,network,current_q,tau)
    batch_indices = np.arange(batch_size)

    delta_mat = np.zeros((batch_size, network.num_actions))
    delta_mat[batch_indices, actions] = delta_vec

    td_update = network.get_TD_update(states, delta_mat)

    weights = optimizer.update_weights(network.get_weights(), td_update)

    network.set_weights(weights)

class Agent(BaseAgent):
    def __init__(self):
        self.name = "expected_sarsa_agent"

    def agent_init(self, agent_config):
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config'])
        self.optimizer = Adam(self.network.layer_sizes, agent_config["optimizer_config"])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0


    def policy(self,state):

        action_values = self.network.get_action_values(state)
        prob_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p= prob_batch.squeeze())
        return action

    def agent_start(self, state):
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):

        self.sum_rewards += reward
        self.episode_steps += 1

        state = np.array([state])
        action = self.policy(state)

        self.replay_buffer.append(self.last_state,self.last_action,reward,0,state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)

            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()

                optimize_network(experiences,self.discount, self.optimizer, self.network, current_q, self.tau)

        self.last_action = action
        self.last_state = state

        return action

    def agent_end(self, reward):
        self.sum_rewards += reward
        self.episode_steps+= 1

        state = np.zeros_like(self.last_state)

        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)

            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()

                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    rl_glue = RLGlue(environment, agent)

    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"],
                                 experiment_parameters["num_episodes"]))

    env_info = {}

    agent_info = agent_parameters

    # one agent setting
    for run in range(1, experiment_parameters["num_runs"] + 1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)

        for episode in tqdm(range(1, experiment_parameters["num_episodes"] + 1)):
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])

            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward
    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)
    shutil.make_archive('results', 'zip', 'results')

experiment_parameters = {
    "num_runs": 1,
    "num_episodes": 350,
    # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after
    # some number of timesteps. Here we use the default of 1000.
    "timeout": 1000
}

# Environment parameters
environment_parameters = {}

current_env = MountainCarEnvironment

# Agent parameters
agent_parameters = {
    'network_config': {
        'state_dim': 2,
        'num_hidden_units': 128,
        'num_actions': 3
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9,
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 50000,
    'minibatch_sz': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}
current_agent = Agent

# run experiment
run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)

plot_result(["expected_sarsa_agent", "random_agent"])

