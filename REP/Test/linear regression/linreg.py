import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#import data
x, y = np.loadtxt('DATA_test_BARTEN.csv', delimiter = ';', unpack = True)

#fit LinearRegression model
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  

#graph output
fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.xlabel("LUMINANCE [cd/m^2]")
plt.ylabel("PUPIL DIAMETER [mm]")
plt.title('Linear Regression Example')

plt.show()
