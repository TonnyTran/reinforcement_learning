import matplotlib.pyplot as plt
from xlrd import open_workbook

# book = open_workbook('../results/result_v0.9.2.2.xls')
# book = open_workbook('../results/DDQN_result_v0.7_gamma0.8_const.xls')
# book = open_workbook('../results/Dueling_DDQN_result_v0.7_gamma0.8_2.xls')
book = open_workbook('../results/QL_result_v0.9.2.9.xls')

discounted_factor = 0.9
nb_steps = 2000000
max_episode_step = 200
interval = 10

sheet = book.sheet_by_index(1)
# read header values into the list
X, R, Q, Q_average = [], [], [], []
for row_index in xrange(0, nb_steps):
    r = sheet.cell_value(row_index % 60000 + 1, row_index / 60000)
    R.append(float(r))


for index in range(0, nb_steps - max_episode_step):
    for i_future in range(1, max_episode_step):
        R[index] += discounted_factor**i_future * R[index+i_future]

for i_episode in range(0, nb_steps/max_episode_step - 1):
    q_episode = 0
    for i_step in range(0, max_episode_step):
        q_episode +=R[i_episode*max_episode_step + i_step]
    Q.append(q_episode)


for ave_index in range(0, len(Q)/interval-1):
    Q_ave = sum(Q[ave_index*interval:((ave_index+1)*interval-1)])/interval
    Q_average.append(Q_ave)
    X.append(ave_index)
print(X)
print(R)
plt.xlabel('x10 episode')
plt.ylabel('Expected reward')
plt.plot(X, Q_average, 'r', label="Expected Reward", zorder=10)
plt.legend()
plt.show()