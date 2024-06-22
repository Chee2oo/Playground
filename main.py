import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

# Beispiel-Daten erstellen
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Daten in ein Pandas DataFrame laden
data = pd.DataFrame(np.c_[X, y], columns=["X", "y"])

# Daten visualisieren
plt.scatter(data["X"], data["y"])
plt.xlabel("X")
plt.ylabel("y")
plt.title("Streudiagramm der Daten")
plt.show()

# Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Startzeit erfassen
start_time = time.time()

# Lineares Regressionsmodell erstellen und trainieren
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Endzeit erfassen
end_time = time.time()

# Vorhersagen auf den Testdaten machen
y_pred = lin_reg.predict(X_test)

# Modellleistung messen
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Laufzeit berechnen
duration = end_time - start_time
print(f"Trainingsdauer: {duration} Sekunden")

# Ergebnisse visualisieren
plt.scatter(X_test, y_test, color="black", label="Testdaten")
plt.plot(X_test, y_pred, color="blue", linewidth=2, label="Vorhersagen")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Lineare Regression")
plt.show()
