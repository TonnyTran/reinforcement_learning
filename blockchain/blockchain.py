"""

"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from action_space import ActionSpace
from discrete import Discrete

class BlockchainEnv(gym.Env):

    def __init__(self):
        self.action_space = ActionSpace(3)
        self.observation_space = spaces.Tuple((Discrete(100), Discrete(100), Discrete(100)))

        # self.seed()
        self.viewer = None
        self.state = None

        self.market_value = 100
        self.alpha = -0.05
        self.ob = 0.1
        self.os = 0.15

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        state_list = list(state)

        action = min(action,state_list[0])

        actions = np.array([action, self.action_space.sample(), self.action_space.sample()])
        for index in range(len(state)):
            win_prob = state[index]*1.0/sum(state)
            if(win_prob > np.random.rand(1)):
                state_list[index] = state_list[index] - actions[index] + 1
            else:
                state_list[index] = state_list[index] - actions[index]

        state = tuple(state_list)
        self.state = state
        self.market_value += sum(actions) * self.alpha
        if (action > 0):    #selling
            reward = action * self.market_value - self.ob
        elif (action < 0):
            reward = action * self.market_value - self.os
        else:
            reward = 0
        done = sum(state)==0
        done = bool(done)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.observation_space.sample()
        print(self.state)
        self.steps_beyond_done = None
        self.market_value = 100
        return np.array(self.state)



    def render(self, mode='human', close=False):
       return

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

class MABlockchainEnv(gym.Env):

    def __init__(self):
        self.max_stake = 100
        self.action_space = spaces.Tuple((ActionSpace(3), ActionSpace(3), ActionSpace(3)))
        self.observation_space = spaces.Tuple((Discrete(self.max_stake), Discrete(self.max_stake), Discrete(self.max_stake)))

        # self.seed()
        self.viewer = None
        self.state = None

        self.market_value = 1
        self.alpha = -0.01
        self.alphaX = -0.01
        self.ob = 0.005
        self.os = 0.006
        self.list_v = []

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, listActions):
        assert self.action_space.contains(listActions), "%r (%s) invalid"%(listActions, type(listActions))
        state = self.state
        listStates = list(state)
        newListStates = list(state)
        newListAction = [None] * len(self.state)
        listRewards = [None] * len(self.state)

        for index in range(len(state)):
            action = min(listActions[index], listStates[index])
            action = max(action, (listStates[index] - self.max_stake + 1))
            newListAction[index] = action

        self.market_value = sum(newListAction) * self.alphaX + 1
        # self.market_value = max(sum(listActions) * self.alpha + self.market_value, 50)
        self.list_v.append(self.market_value)

        for index in range(len(state)):
            # # no mining
            # newListStates[index] = listStates[index] - newListAction[index]

            # mining progress
            if(sum(state) != 0):
                win_prob = state[index]*1.0/sum(state)
            else:
                win_prob = 0

            if(win_prob > np.random.rand(1)):
                newListStates[index] = listStates[index] - newListAction[index] + 1
            else:
                newListStates[index] = listStates[index] - newListAction[index]

        for index in range(len(state)):
            action = min(listActions[index], listStates[index])
            action = max(action, (listStates[index] - self.max_stake + 1))
            newListAction[index] = action

            # caculate the reward
        for index in range(len(state)):
            if (newListAction[index] > 0):  # selling
                listRewards[index] = newListAction[index] * self.market_value - self.ob
            elif (newListAction[index] < 0):
                listRewards[index] = newListAction[index] * self.market_value - self.os
            else:
                listRewards[index] = 0

        state = tuple(newListStates)
        self.state = state

        # done = sum(state)==0
        # done = bool(done)
        done = False

        return np.array(self.state), listRewards, done, {}

    def reset(self):
        print(self.state)
        self.state = self.observation_space.sample()
        print(self.state)
        self.steps_beyond_done = None
        self.market_value = 100
        return np.array(self.state)


    def render(self, mode='human', close=False):
       return
