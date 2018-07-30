from rl.agents.dqn import DQNAgent, AbstractDQNAgent
import warnings
from copy import deepcopy

import numpy as np
import xlwt
from keras.callbacks import History

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

class ListDQNAgents(AbstractDQNAgent):
    def __init__(self, nb_agents=3, nb_actions=None, memory=None, processor=None, nb_steps_warmup=100, version=None,
                 anneal_steps=None, target_model_update=1e-2, policy=None):

        # vesion
        self.version = version
        # vary epsilon greedy policy
        self.vary_eps = True
        self.listDQNAgents = [None] * nb_agents

        # eGreedy parameters
        self.init_exp = 0.9
        self.final_exp = 0.0
        self.exploration = self.init_exp
        self.anneal_steps = anneal_steps

        for index in range(nb_agents):
            model = Sequential()
            model.add(Flatten(input_shape=(1, 3)))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(nb_actions, activation='linear'))

            print(model.summary())

            self.listDQNAgents[index] = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, processor=processor,
                nb_steps_warmup=nb_steps_warmup, target_model_update=target_model_update, policy=policy,
                enable_double_dqn=False, enable_dueling_network=False)

        # Parameters.
        self.nb_agents = nb_agents
        self.nb_actions = nb_actions
        self.nb_steps_warmup = nb_steps_warmup
        self.target_model_update = target_model_update

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

        self.processor = processor
        self.training = False
        self.step = 0

        # from AbstractDQNAgent
        self.memory_interval = 1
        self.train_interval = 1
        self.batch_size = 32


    def compile(self, optimizer=None, metrics=[]):
        for dqn in self.listDQNAgents:
            dqn.compile(optimizer, metrics)
        self.compiled = True


    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
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
        # eGreedy parameters
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True
        self.nb_steps = nb_steps
        print(self.nb_steps)

        # open workbook to store result
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet('DQN')
        # sheet_step = workbook.add_sheet('step')


        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)

        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }

        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.zeros(self.nb_agents, dtype=np.float32)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

                    # # Perform random starts at beginning of episode and do not record them into the experience.
                    # # This slightly changes the start position between games.
                    # nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    # for _ in range(nb_random_start_steps):
                    #     if start_step_policy is None:
                    #         listActions = env.action_space.sample()
                    #     else:
                    #         listActions = start_step_policy(observation)
                    #     if self.processor is not None:
                    #         action = self.processor.process_action(action)
                    #     callbacks.on_action_begin(action)
                    #     observation, reward, done, info = env.step(action)
                    #     observation = deepcopy(observation)
                    #     if self.processor is not None:
                    #         observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                    #     callbacks.on_action_end(action)
                    #     if done:
                    #         warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    #         observation = deepcopy(env.reset())
                    #         if self.processor is not None:
                    #             observation = self.processor.process_observation(observation)
                    #         break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                listActions = self.forward(observation)
                if self.processor is not None:
                    listActions = self.processor.process_action(listActions)

                reward = np.zeros(self.nb_agents, dtype=np.float32)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(listActions)
                    observation, r, done, info = env.step(listActions)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(listActions)
                    for index in range(self.nb_agents):
                        reward[index] += r[index]

                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': listActions,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)

                # # step reward
                # sheet_step.write(self.step + 1, 0, str(self.step))
                # sheet_step.write(self.step + 1, 1, str(episode_reward[0]))
                # sheet_step.write(self.step + 1, 2, str(episode_reward[1]))
                # sheet_step.write(self.step + 1, 3, str(episode_reward[2]))
                # sheet_step.write(self.step + 1, 4, str(sum(episode_reward)))

                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

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
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()
        # close file
        # file_out.close()
        file_name = 'result_v' + self.version + '.xls'
        if (self.listDQNAgents[0].enable_double_dqn):
            file_name = 'DDQN_' + file_name
        if (self.listDQNAgents[0].enable_dueling_network):
            file_name = 'Dueling_' + file_name
        workbook.save('../results/' + file_name)

        # print market value
        # print(env.list_v)
        return history

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            for index in range(self.nb_agents):
                self.listDQNAgents[index].model.reset_states()
                self.listDQNAgents[index].target_model.reset_states()


    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        listActions = [None] * self.nb_agents
        for index in range(self.nb_agents):
            q_values = self.listDQNAgents[index].compute_q_values(state)
            if self.training:
                if (self.vary_eps):
                    self.annealExploration()
                    action = self.listDQNAgents[index].policy.select_action_vary(q_values=q_values, eps=(self.exploration))
                else:
                    action = self.listDQNAgents[index].policy.select_action(q_values=q_values)
            else:
                action = self.listDQNAgents[index].test_policy.select_action(q_values=q_values)
            listActions[index] = action

            # Book-keeping.
        self.recent_observation = observation
        self.recent_action = listActions

        return tuple(listActions)

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
        metrics = [np.nan for _ in self.listDQNAgents[0].metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            for index in range(self.nb_agents):
                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = []
                reward_batch = []
                action_batch = []
                terminal1_batch = []
                state1_batch = []
                for e in experiences:
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward[index])
                    action_batch.append(e.action[index])
                    terminal1_batch.append(0. if e.terminal1 else 1.)

                # Prepare and validate parameters.
                state0_batch = self.listDQNAgents[index].process_state_batch(state0_batch)
                state1_batch = self.listDQNAgents[index].process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                assert reward_batch.shape == (self.listDQNAgents[index].batch_size,)
                assert terminal1_batch.shape == reward_batch.shape
                assert len(action_batch) == len(reward_batch)

                # Compute Q values for mini-batch update.
                if self.listDQNAgents[index].enable_double_dqn:
                    # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                    # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                    # while the target network is used to estimate the Q value.
                    q_values = self.listDQNAgents[index].model.predict_on_batch(state1_batch)
                    assert q_values.shape == (self.listDQNAgents[index].batch_size, self.nb_actions)
                    actions = np.argmax(q_values, axis=1)
                    assert actions.shape == (self.listDQNAgents[index].batch_size,)

                    # Now, estimate Q values using the target network but select the values with the
                    # highest Q value wrt to the online model (as computed above).
                    target_q_values = self.listDQNAgents[index].target_model.predict_on_batch(state1_batch)
                    assert target_q_values.shape == (self.listDQNAgents[index].batch_size, self.nb_actions)
                    q_batch = target_q_values[range(self.listDQNAgents[index].batch_size), actions]
                else:
                    # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                    # We perform this prediction on the target_model instead of the model for reasons
                    # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                    target_q_values = self.listDQNAgents[index].target_model.predict_on_batch(state1_batch)
                    assert target_q_values.shape == (self.listDQNAgents[index].batch_size, self.nb_actions)
                    q_batch = np.max(target_q_values, axis=1).flatten()
                assert q_batch.shape == (self.listDQNAgents[index].batch_size,)

                targets = np.zeros((self.listDQNAgents[index].batch_size, self.nb_actions))
                dummy_targets = np.zeros((self.listDQNAgents[index].batch_size,))
                masks = np.zeros((self.listDQNAgents[index].batch_size, self.nb_actions))

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.listDQNAgents[index].gamma * q_batch
                # Set discounted reward to zero for all states that were terminal.
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                Rs = reward_batch + discounted_reward_batch
                for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                    target[action] = R  # update action with estimated accumulated reward
                    dummy_targets[idx] = R
                    mask[action] = 1.  # enable loss for this specific action
                targets = np.array(targets).astype('float32')
                masks = np.array(masks).astype('float32')

                # Finally, perform a single update on the entire batch. We use a dummy target since
                # the actual loss is computed in a Lambda layer that needs more complex input. However,
                # it is still useful to know the actual target to compute metrics properly.
                ins = [state0_batch] if type(self.listDQNAgents[index].model.input) is not list else state0_batch
                metrics = self.listDQNAgents[index].trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
                metrics = [metric for idx, metric in enumerate(metrics) if
                           idx not in (1, 2)]  # throw away individual losses
                metrics += self.listDQNAgents[index].policy.metrics
                if self.processor is not None:
                    metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            for index in range(self.nb_agents):
                self.listDQNAgents[index].update_target_model_hard()

        return metrics

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=500, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
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
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += sum(r)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

    def annealExploration(self, stategy='linear'):
        ratio = max((self.anneal_steps - self.step) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def save_weights(self, ENV_NAME, overwrite=False):
        for index in range(self.nb_agents):
            self.listDQNAgents[index].save_weights('dqn_{}_{}_weights.h5f'.format(index, ENV_NAME), overwrite=True)
