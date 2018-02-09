import numpy as np
import matplotlib.pyplot as plt

def rbf_iter(X, MU, sigma):
    fi = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(MU)):
            fi[i,j] = np.exp((-(X[i]-MU[j])**2)/(2*sigma[j]**2))
    return fi

def rbf(x, mu, sigma):
    fi = np.exp((-(x-mu)**2)/(2*sigma**2))
    return fi

rbf_vect = np.vectorize(rbf)

def rbf_learning(pattern, target):
    weight = np.random.normal(size=pattern.shape)
    mu = np.random.normal(size=pattern.shape)
    sigma = np.ones(pattern.shape)

    #fi = rbf_vect(pattern, mu, sigma.T)
    fi2 = rbf_iter(pattern, mu, sigma)
    #total_error = np.abs(np.dot(fi, weight)-target)**2

    weight = np.linalg.solve(np.dot(fi2.T, fi2), np.dot(fi2.T, target))
    predict = np.dot(fi2, weight)
    residual = target - predict
    return predict, residual

pattern = np.arange(0, 2*np.pi, 0.1)
pattern = pattern.reshape((-1,1))
target_sin = np.sin(2*pattern)
target_square = np.sign(target_sin)

predict, residual = rbf_learning(pattern, target_sin)

fig = plt.figure()
plt.plot(target_sin)
plt.plot(predict)
plt.show()
