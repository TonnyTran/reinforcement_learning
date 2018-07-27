import matplotlib.pyplot as plt
from xlrd import open_workbook

book = open_workbook('../results/result_v1.0.0.xls')
# book = open_workbook('../results/DDQN_result_v0.7_gamma0.8_const.xls')
# book = open_workbook('../results/Dueling_DDQN_result_v0.7_gamma0.8_2.xls')
# book = open_workbook('../results/QL_result_v0.9.2.9.xls')

sheet = book.sheet_by_index(0)
discounted_factor = 0.9
nb_steps = 2000000
max_episode_step = 200
interval = 100
# read header values into the list
# DDQNkeys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
X, R, R_average = [], [], []
for row_index in xrange(1, sheet.nrows):
    # x = sheet.cell_value(row_index, 0)
    y = sheet.cell_value(row_index, 4)
    R.append(float(y))

for ave_index in range(0, len(R)/interval-1):
    R_ave = sum(R[ave_index*interval:((ave_index+1)*interval-1)])/interval
    R_average.append(R_ave)
    X.append(ave_index)

# print(X)
# print(R)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.plot(X, R_average, 'r', label="DQN", zorder=10)
plt.legend()
plt.show()