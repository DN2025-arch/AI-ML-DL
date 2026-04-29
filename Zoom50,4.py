import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

data = pd.read_csv(r"Zooms\ML and AI\Zoom50,3CSV.csv")
print(data.head())


x = data[["X"]]
y = data["Y"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
rmse_lin = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE (Linear Regression): {rmse_lin:.4f}")
