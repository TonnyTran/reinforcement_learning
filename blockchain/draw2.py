import matplotlib.pyplot as plt
from xlrd import open_workbook

book = open_workbook('../results/result_v0.9.2.3.xls')
book2 = open_workbook('../results/QL_result_v0.9.2.3.xls')
sheet = book.sheet_by_index(0)
sheet2 = book2.sheet_by_index(0)

# read header values into the list
# DDQNkeys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
X, Y, Y2, Y3 = [], [], [], []
for row_index in xrange(1, sheet.nrows):
    x = sheet.cell_value(row_index, 0)
    y = sheet.cell_value(row_index, 4)
    y2 = sheet2.cell_value(row_index, 4)
    X.append(int(x))
    Y.append(float(y))
    Y2.append(float(y2))
print(X)
print(Y)
print(Y2)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.plot(X, Y, 'r', label="DQN", zorder=10)
plt.plot(X, Y2, 'b', label="QL", zorder=10)
plt.legend()
plt.show()