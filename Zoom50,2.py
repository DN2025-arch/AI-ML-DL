import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

california = fetch_california_housing() # Form of CSV
df = pd.DataFrame(california.data, columns=california.feature_names)
df["MedHouseVal"] = california.target

print(df.head())

x = df[["MedInc","AveRooms"]]
y = df["MedHouseVal"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

lin_model = LinearRegression()
lin_model.fit(x_train, y_train)
y_pred_lin = lin_model.predict(x_test)
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print(f"RMSE (Linear Regression): {rmse_lin:.4f}")

poly = PolynomialFeatures(degree=2) # Degree: Amount of Training
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_train_poly,y_train)
y_pred_poly = poly_model.predict(x_test_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test,y_pred_poly))
print(f"RMSE (Polynomial Regression): {rmse_poly:.4f}")


