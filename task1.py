import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

np.random.seed(42)
n = 20640
MedInc = np.random.lognormal(1.5, 0.7, n).clip(0.5, 15)
HouseAge = np.random.uniform(1, 52, n)
AveRooms = np.random.lognormal(1.6, 0.4, n).clip(1, 20)
AveBedrms = (AveRooms * np.random.uniform(0.15, 0.35, n)).clip(1, 6)
Population = np.random.lognormal(6.5, 1.0, n).clip(3, 35682)
AveOccup = np.random.lognormal(1.1, 0.5, n).clip(1, 20)
Latitude = np.random.uniform(32.5, 42.0, n)
Longitude = np.random.uniform(-124.4, -114.3, n)
noise = np.random.normal(0, 0.3, n)
MedHouseVal = (0.45*MedInc + 0.003*HouseAge + 0.08*AveRooms
    - 0.06*AveBedrms - 0.00001*Population
    - 0.04*AveOccup - 0.05*Latitude
    + 0.01*Longitude + 1.8 + noise).clip(0.15, 5.0)

df = pd.DataFrame({'MedInc':MedInc,'HouseAge':HouseAge,
    'AveRooms':AveRooms,'AveBedrms':AveBedrms,
    'Population':Population,'AveOccup':AveOccup,
    'Latitude':Latitude,'Longitude':Longitude,
    'MedHouseVal':MedHouseVal})

X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('='*40)
print('  MODEL EVALUATION METRICS')
print('='*40)
print(f'  MAE  : {mae:.4f}')
print(f'  RMSE : {rmse:.4f}')
print(f'  R2   : {r2:.4f}')
print('='*40)

joblib.dump(model, 'linear_regression_model.pkl')
print('Model saved!')

plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=6)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, 'r--', lw=1.5)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
print('Chart saved!')
