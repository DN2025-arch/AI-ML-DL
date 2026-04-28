


def findMean(x):
    return sum(x)/len(x)

x = [1,2,4,3,5]
y = [1,3,3,2,5]

meanX = findMean(x)
meanY = findMean(y)

num = 0
den = 0
for i in range(len(x)):
    num = num + ((x[i] - meanX) * (y[i] - meanY))
    den = den + pow((x[i] - meanX), 2)

m = num / den
c = round(meanY - m * meanX, 1)
print("m =", m)
print("c =", c)

# m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi-mean(x))^2)

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5]])
y = np.array([[1],[3],[3],[2],[5]])

reg = LinearRegression().fit(x,y)
print("m =", reg.coef_)
print("c =", reg.intercept_)

plt.scatter(x,y, color="blue", label="Data Points")
plt.plot(x, reg.predict(x), color="red", label="Best Line Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
