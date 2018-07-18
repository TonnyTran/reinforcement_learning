import matplotlib.pyplot as plt
from xlrd import open_workbook

book = open_workbook('../results/result_v0.4.xls')
# book = open_workbook('../results/DDQN_result.xls')
# book = open_workbook('../results/Dueling_DDQN_result0.2.xls')

sheet = book.sheet_by_index(0)
# read header values into the list
# DDQNkeys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
X, Y = [], []
for row_index in xrange(1, sheet.nrows):
    x = sheet.cell_value(row_index, 0)
    y = sheet.cell_value(row_index, 4)
    X.append(int(x))
    Y.append(float(y))
print(X)
print(Y)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.plot(X, Y, 'r', label="DQN", zorder=10)
plt.legend()
plt.show()