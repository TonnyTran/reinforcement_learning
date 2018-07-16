import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from blockchain import BlockchainEnv

class BlockchainProcessor(Processor):
    def process_action(self, action):
        action = action - (nb_actions / 2)
        return action

ENV_NAME = 'BlockChain'

# Get the environment and extract the number of actions.
env = BlockchainEnv()
np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.nb_actions

# Next, we build a very simple model.

model = Sequential()
model.add(Flatten(input_shape=(1, 3)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
processor = BlockchainProcessor()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, processor=processor, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=10000, visualize=True, verbose=2, log_interval=1000, nb_max_episode_steps=500)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=50, visualize=True, nb_max_episode_steps=100)
