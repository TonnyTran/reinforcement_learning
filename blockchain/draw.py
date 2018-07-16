import matplotlib.pyplot as plt
from xlrd import open_workbook

book = open_workbook('result.xls')
book2 = open_workbook('DDQN_result.xls')
book3 = open_workbook('Dueling_DDQN_result.xls')
sheet = book.sheet_by_index(0)
sheet2 = book2.sheet_by_index(0)
sheet3 = book3.sheet_by_index(0)

# read header values into the list
# DDQNkeys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
X, Y, Y2, Y3 = [], [], [], []
for row_index in xrange(1, sheet.nrows):
    x = sheet.cell_value(row_index, 0)
    y = sheet.cell_value(row_index, 4)
    y2 = sheet2.cell_value(row_index, 4)
    y3 = sheet3.cell_value(row_index, 4)
    X.append(int(x))
    Y.append(float(y))
    Y2.append(float(y2))
    Y3.append(float(y3))
print(X)
print(Y)
print(Y2)
print(Y3)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.plot(X, Y, 'r', label="DQN", zorder=10)
plt.plot(X, Y2, 'b', label="DDQN", zorder=10)
plt.plot(X, Y3, 'g', label="D3QN", zorder=10)
plt.legend()
plt.show()