import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from math import log10
import pandas as pd

#data import
data = pd.read_csv("DATA_imp.csv")

#assign data to variables
train = data.sample(frac = 0.3)
test = data.drop(train.index)

x_LH = train['Luminance']
y_LH = train['moon spencer']
test_x_LH = test['Luminance']
test_y_LH = test['moon spencer']

X_test = np.array([log10(x) if x != 0 else 0 for x in test_x_LH.values]).reshape((-1, 1)) 
X = np.array([log10(x) if x != 0 else 0  for x in x_LH.values]).reshape((-1, 1))
y = y_LH.values


#fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=0.1,
               coef0=1)

#graph output
lw = 1
svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['DODGERBLUE', 'RED', 'ORANGE']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,7), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].scatter(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor='lightsteelblue', s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=False, shadow=True)

fig.text(0.5, 0.04, 'light intensity [log blondel units]', ha='center', va='center')
fig.text(0.06, 0.5, 'pupil diameter [mm]', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=10)
plt.show()
