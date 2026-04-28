import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

data = pd.read_csv("Zooms\ML and AI\iris_data.csv")

x = data[["sepal_length","sepal_width","petal_width"]]
y = data[["petal_length"]]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

poly = PolynomialFeatures(degree=2)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
predictions = poly_model.predict(x_test_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse_poly:.4f}")

plt.scatter(y_test,predictions, color="blue")
#plt.plot(y_test,predictions, color="red")
plt.show()