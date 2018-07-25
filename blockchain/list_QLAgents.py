from rl.agents.dqn import DQNAgent, AbstractDQNAgent
import warnings
from copy import deepcopy

import numpy as np
import xlwt
from keras.callbacks import History
from rl.agents.tabular_q_learner import QLearner

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import math

class ListQLAgents():
    def __init__(self, nb_agents=3, state_dim=None, nb_actions=None, processor=None, policy=None):

        # vesion
        self.version = '0.9.2.4'
        # vary epsilon greedy policy
        self.vary_eps = True
        self.listDQNAgents = [None] * nb_agents

        for index in range(nb_agents):
            self.listDQNAgents[index] = QLearner(state_dim, nb_actions)

        # Parameters.
        self.nb_agents = nb_agents
        self.nb_actions = nb_actions


        self.processor = processor
        self.training = False
        self.step = 0

    def fit(self, env, nb_steps, nb_max_episode_steps=None):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """

        self.training = True
        self.nb_steps = nb_steps

        # open workbook to store result
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet('DQN')
        # sheet_step = workbook.add_sheet('step')

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    episode_step = np.int16(0)
                    episode_reward = np.zeros(self.nb_agents, dtype=np.float32)

                    # initialize
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    state = self.processor.digitalizeState(observation=observation, env=env)
                    listActions = self.initializeState(state)
                    if self.processor is not None:
                        listActions = self.processor.process_action(listActions)
                    assert observation is not None


                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                reward = np.zeros(self.nb_agents, dtype=np.float32)
                done = False
                observation, r, done, _ = env.step(listActions)
                observation = deepcopy(observation)
                # if self.processor is not None:
                #     observation, r, done, info = self.processor.process_step(observation, r, done, info)
                state = self.processor.digitalizeState(observation=observation, env=env)
                reward += r
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                listActions = self.updateModel(state, r)
                if self.processor is not None:
                    listActions = self.processor.process_action(listActions)
                episode_reward += reward

                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.

                    # print("Episode {}".format(episode))
                    # print("Reward for this episode: {}".format(episode_reward))
                    # print("Average reward for current episode: {}".format(episode_reward/episode_step))

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode':episode,
                        'nb_steps': self.step,
                        'episode_reward': sum(episode_reward),
                        'mean_reward': sum(episode_reward)/episode_step
                    }
                    print(episode_logs)
                    # file_out.write(episode_reward)
                    sheet.write(episode + 1, 0, str(episode))
                    sheet.write(episode + 1, 1, str(episode_reward[0]))
                    sheet.write(episode + 1, 2, str(episode_reward[1]))
                    sheet.write(episode + 1, 3, str(episode_reward[2]))
                    sheet.write(episode + 1, 4, str(sum(episode_reward)))

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True

        file_name = 'QL_result_v' + self.version + '.xls'
        workbook.save('../results/' + file_name)

        # print market value
        # print(env.list_v)
        return

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None

    def initializeState(self, state):
        listActions = [None] * self.nb_agents
        for i_agent in range(0, self.nb_agents):
            listActions[i_agent] = self.listDQNAgents[i_agent].initializeState(state)
        return listActions

    def updateModel(self, state, reward):
        listActions = [None] * self.nb_agents
        for i_agent in range(0, self.nb_agents):
            listActions[i_agent] = self.listDQNAgents[i_agent].updateModel(state, reward[i_agent])
        return listActions