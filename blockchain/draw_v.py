import matplotlib.pyplot as plt
from xlrd import open_workbook

book = open_workbook('../data_NEO/Bittrex_NEOUSD_1h.xls')
# book = open_workbook('../results/DDQN_result_v0.7_gamma0.8_const.xls')
# book = open_workbook('../results/Dueling_DDQN_result_v0.7_gamma0.8_2.xls')

sheet = book.sheet_by_index(0)
# read header values into the list
# DDQNkeys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
X, Y = [], []
interval = 4
for index in xrange(1, sheet.nrows/interval):
    row_index = sheet.nrows - (index * interval)
    x = index
    y = round(sheet.cell_value(row_index, 2)/2) * 2
    X.append(int(x))
    Y.append(float(y))
print(max(Y))
print(min(Y))
plt.xlabel('Number of episodes')
plt.ylabel('Price')
plt.plot(X, Y, 'r', label="Price(USD)", zorder=10)
plt.legend()
plt.show()

def normalizeValue(listValue):
    minValue = min(listValue)
    maxValue = max(listValue)
    valueInterval = 2
    newListValue = []

    for index in range(0, len(listValue)):
        newListValue[index] = (listValue[index] - minValue) / valueInterval
    return listValue