import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_linear_pred = linear_model.predict(X)
y_linear_test_pred = linear_model.predict(X_test)

poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
X_poly = poly_features.transform(X)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_poly_pred = poly_model.predict(X_poly)
y_poly_test_pred = poly_model.predict(X_test_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, s=15, label="Data")
plt.plot(X, y_linear_pred, color="red", label="Linear Regression")
plt.plot(X, y_poly_pred, color="green", label="Polynomial Regression (degree=3)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
mse_linear = mean_squared_error(y_test, y_linear_test_pred)
mse_poly = mean_squared_error(y_test, y_poly_test_pred)

print(f"Linear Regression MSE: {mse_linear:.4f}")
print(f"Polynomial Regression MSE: {mse_poly:.4f}")
