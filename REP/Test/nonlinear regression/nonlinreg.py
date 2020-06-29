import numpy as np
import scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

#import data
X, y = np.loadtxt('DATA_test_BARTEN.csv', delimiter = ';', unpack = True)

#define the function
def func(X, a, b, Offset): 
    return 1.0 / (1.0 + np.exp(-a * (X-b))) + Offset

#sum squared error
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") 
    val = func(X, *parameterTuple)
    return np.sum((y - val) ** 2.0)

def generate_Initial_Parameters():
    #values used for bounds
    maxX = max(X)
    minX = min(X)
    maxY = max(y)
    minY = min(y)

    parameterBounds = []
    parameterBounds.append([minX, maxX]) #looks for bounds for a
    parameterBounds.append([minX, maxX]) #looks for bounds for b
    parameterBounds.append([0.0, maxY]) #looks for bounds for Offset

    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

#generate initial parameter values
geneticParameters = generate_Initial_Parameters()

#curve fit the test data
fittedParameters, pcov = curve_fit(func, X, y, geneticParameters)
modelPredictions = func(X, *fittedParameters) 

#graph output
def ModelAndScatterPlot(graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    axes.plot(X, y, 'ro')

    xModel = np.linspace(min(X), max(X))
    yModel = func(xModel, *fittedParameters)

    axes.plot(xModel, yModel, c = 'dodgerblue')

    axes.set_xlabel('LUMINANCE [cd/m^2]') 
    axes.set_ylabel('PUPIL DIAMETER [mm]')
    plt.title('Non-linear Regression Example')
    plt.legend(('Data', 'Non-linear fit'), loc = 'upper right')

    plt.show()
    plt.close('all') 

graphWidth = 800
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)
