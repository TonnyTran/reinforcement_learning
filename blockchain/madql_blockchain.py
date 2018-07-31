import numpy as np
from keras.optimizers import Adam

from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from blockchain import MABlockchainEnv
from list_DQNAgents import ListDQNAgents

class MABlockchainProcessor(Processor):
    def process_action(self, listActions):
        list_actions = list(listActions)
        for index in range(len(list_actions)):
            list_actions[index] = list_actions[index] - (nb_actions / 2)
        return tuple(list_actions)

ENV_NAME = 'BlockChain'
nb_agents = 5

# Get the environment and extract the number of actions.
env = MABlockchainEnv()
np.random.seed(123)
# env.seed(123)
# nb_actions = env.action_space[0].nb_actions
nb_actions = env.nb_actions
# Next, we build a very simple model.

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(0.1)
processor = MABlockchainProcessor()

version = '1.2.0_7miners'
nb_steps = 500000
nb_max_episode_steps = 200
anneal_steps = 400000

listDQNAgents = ListDQNAgents(nb_agents=nb_agents, nb_actions=nb_actions, memory=memory, processor=processor, nb_steps_warmup=100,
               target_model_update=1e-2, version=version, anneal_steps=anneal_steps, policy=policy)

listDQNAgents.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
listDQNAgents.fit(env, nb_steps=nb_steps, visualize=True, verbose=2, log_interval=1000, nb_max_episode_steps=nb_max_episode_steps)

# After training is done, we save the final weights.
listDQNAgents.save_weights(ENV_NAME)

# Finally, evaluate our algorithm for 5 episodes.
listDQNAgents.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=nb_max_episode_steps)
