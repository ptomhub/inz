import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from math import log10
import pandas as pd

#data import
data = pd.read_csv("dataset.csv")

#assign data to variables
train = data.sample(frac = 0.3)
test = data.drop(train.index)

x_LH = train['Luminance']
y_LH = train['Diameter']
test_x_LH = test['Luminance']
test_y_LH = test['Diameter']

X_test = np.array([log10(x) if x != 0 else 0 for x in test_x_LH.values]).reshape((-1, 1)) 
X = np.array([log10(x) if x != 0 else 0  for x in x_LH.values]).reshape((-1, 1))
y = y_LH.values

#SVM kernel of choice
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

#graph output
fig = plt.figure(figsize=(12,7))
axe = fig.add_subplot(111)
model = svr_rbf.fit(X, y)
predictions_for_test = model.predict(X_test)
axe.scatter(X[np.setdiff1d(np.arange(len(X)), svr_rbf.support_)],
                     y[np.setdiff1d(np.arange(len(X)), svr_rbf.support_)],
                     color='orange', s=80,
                     label='Real data')
axe.scatter(X[svr_rbf.support_], y[svr_rbf.support_], facecolor="none",
                     edgecolor='red', s=40,
                     label='{} support vectors'.format('RBF'))
axe.scatter(X_test, predictions_for_test,facecolor = 'none', edgecolor='black', s=60, 
                     label='{} test data'.format('RBF'))
axe.scatter(X, model.predict(X), color='dodgerblue', s=30, 
                    label='{} model'.format('RBF'))
axe.legend(loc=2, bbox_to_anchor=(0.75, 1),
                     ncol=1, fancybox=False, shadow=True)

fig.text(0.5, 0.04, 'light intensity [log blondel units]', ha='center', va='center')
fig.text(0.06, 0.5, 'pupil diameter [mm]', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=10)
fig.show()
plt.show()
