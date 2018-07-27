from __future__ import print_function
from rl.core import Processor
from blockchain import MABlockchainEnv
from list_QLAgents import ListQLAgents

class MABlockchainProcessor(Processor):
    def process_action(self, listActions):
        list_actions = list(listActions)
        for index in range(len(list_actions)):
            list_actions[index] = list_actions[index] - (nb_actions / 2)
        return tuple(list_actions)

    def digitalizeState(self, observation, env=None):
        state=0
        for index in range(0,len(observation)):
            state += observation[index] * ((env.max_stake + 1) ** index)
        return state

ENV_NAME = 'BlockChain'
# Get the environment and extract the number of actions.
env = MABlockchainEnv()

nb_agents = 3
nb_actions = env.nb_actions
state_dim = (env.max_stake + 1) ** nb_agents
processor = MABlockchainProcessor()
print(state_dim)

version = '0.9.2.9'
nb_steps = 2000000
anneal_steps = int(3.0/4 * nb_steps)
nb_max_episode_steps=200
listQLAgents = ListQLAgents(nb_agents=nb_agents, state_dim=state_dim, nb_actions=nb_actions, anneal_steps=anneal_steps, processor=processor, version=version)
listQLAgents.fit(env=env, nb_steps=nb_steps, nb_max_episode_steps=nb_max_episode_steps)

