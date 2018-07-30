import numpy as np

state = np.array([1, 2, 3], np.int8)
i = sum(state[0:2])
j = state[1]
prob = state[1]*1.0/i
t = np.random.rand(1)
if(t < prob):
    print("ngon")
print(prob)

episode_reward = np.zeros(3, dtype=np.float32)
print(episode_reward)