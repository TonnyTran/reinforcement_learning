import matplotlib.pyplot as plt
from xlrd import open_workbook
import numpy as np

class MarketValue():
    def __init__(self):
        self.valueInterval =2  # 2,4,6
        self.timeInterval = 4  # hours
        self.listValue = []
        book = open_workbook('../data_NEO/Bittrex_NEOUSD_1h.xls')
        sheet = book.sheet_by_index(0)

        for index in xrange(1, sheet.nrows / self.timeInterval):
            row_index = sheet.nrows - (index * self.timeInterval)
            value = round(sheet.cell_value(row_index, 2) / self.valueInterval) * self.valueInterval
            self.listValue.append(int(value))

        self.maxValue = max(self.listValue)
        self.minValue = min(self.listValue)
        self.numberValue = (self.maxValue - self.minValue) / self.valueInterval + 1
        self.listValueNormalized = self.normalizeValues(self.listValue)
        self.valueMatrixTranfer = self.calculateTranferProbability(self.listValueNormalized)

        self.currentValueNormalized = np.random.randint(self.numberValue)
        self.currentValue = self.denormalizeValue(self.currentValueNormalized)


    def normalizeValues(self, listValue):
        newListValue = []
        for index in range(0, len(listValue)):
            newListValue.append((listValue[index] - self.minValue) / self.valueInterval)
        return newListValue

    def normalizeValue(self, value):
        newValue = (value - self.minValue) / self.valueInterval
        return newValue

    def denormalizeValue(self, valueNomalized):
        value = valueNomalized * self.valueInterval + self.minValue
        return value

    def calculateTranferProbability(self, listValueNormalized):
        valueMatrix = np.zeros((self.numberValue, self.numberValue))
        for index in range(0, len(listValueNormalized) - 1):
            valueMatrix[listValueNormalized[index],listValueNormalized[index+1]] += 1

        valueMatrixTranfer = np.zeros((self.numberValue, self.numberValue))
        for rowIndex in range(self.numberValue):
            sumRow = sum(valueMatrix[rowIndex, :])
            for colIndex in range(self.numberValue):
                valueMatrixTranfer[rowIndex, colIndex] = valueMatrix[rowIndex, colIndex] / sumRow
        return valueMatrixTranfer

    def step(self):
        ran = np.random.rand()
        sumRow = 0
        for colIndex in range(0, self.numberValue):
            sumRow += self.valueMatrixTranfer[self.currentValueNormalized, colIndex]
            if (sumRow > ran):
                self.currentValueNormalized = colIndex
                break
        self.currentValue = self.denormalizeValue(self.currentValueNormalized)

    def reset(self):
        # self.currentValueNormalized = np.random.randint(self.numberValue)
        self.currentValueNormalized = 15
        self.currentValue = self.denormalizeValue(self.currentValueNormalized)

    def drawValue(self, listValue):
        X = []
        for index in range(0, len(listValue)):
            X.append(index)
        plt.xlabel('Number of steps')
        plt.ylabel('Price')
        plt.plot(X, listValue, 'r', label="Price(USD)", zorder=10)
        plt.legend()
        plt.show()

# marketValue = MarketValue()
# # marketValue.drawValue(marketValue.listValue)
# print(len(marketValue.listValue))
# print(marketValue.valueMatrixTranfer.shape)
# print(marketValue.valueMatrixTranfer)
# listValue = []
# for i in range(0,500):
#     marketValue.step()
#     listValue.append(marketValue.currentValue)
# marketValue.drawValue(listValue)
