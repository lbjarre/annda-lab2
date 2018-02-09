import numpy as np
import matplotlib.pyplot as plt

pattern = np.arange(0, 2*np.pi, 0.1)

targetSin = np.sin(2*pattern)
targetSquare = np.sign(targetSin)

def rbf(pattern, mu):
    sigma = 1
    fi = np.exp((-(x-mu)**2)/(2*sigma**2))
    return fi

def RBFLearning(pattern, target):
    weight = np.random.normal(size=pattern)
    mu = []
    fi = np.vectorize(rbf(pattern, mu))
