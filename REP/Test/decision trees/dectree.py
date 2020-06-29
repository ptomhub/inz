import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from math import log10

#data import
data = pd.read_csv("DATA_imp.csv")

#assign data to variables
train = data.sample(frac = 0.2)
test = data.drop(train.index)

x_LH = train['Luminance']
y_LH = train['blackie howland']
test_x_LH = test['Luminance']
test_y_LH = test['blackie howland']

X_test = np.array([log10(x) if x != 0 else 0 for x in test_x_LH.values]).reshape((-1, 1)) 
X = np.array([log10(x) if x != 0 else 0  for x in x_LH.values]).reshape((-1, 1))
y = y_LH.values

#fit regression model
regr_1 = DecisionTreeRegressor(max_depth=1)
regr_2 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X, y)
regr_2.fit(X, y)

#predict
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

#graph output
plt.figure()
plt.scatter(X, y, s=20, c="red", label="data")
plt.plot(X_test, y_1, color="orange",label="max_depth=1", linewidth=3)
plt.plot(X_test, y_2, color="dodgerblue", label="max_depth=4", linewidth=3)
plt.xlabel("LIGHT INTENSITY [log blondel units]")
plt.ylabel("PUPIL DIAMETER [mm]")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
