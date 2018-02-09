import numpy as np
import matplotlib.pyplot as plt

def rbf_iter(X, MU, sigma):
    fi = np.zeros((len(X), len(MU)))
    for i in range(len(X)):
        for j in range(len(MU)):
            fi[i,j] = np.exp((-(X[i]-MU[j])**2)/(2*sigma[j]**2))
    return fi

def rbf(x, mu, sigma):
    fi = np.exp((-(x-mu)**2)/(2*sigma**2))
    return fi

rbf_vect = np.vectorize(rbf)

def rbf_learning(pattern, target, mu):
    sigma = np.ones(mu.shape)

    #fi = rbf_vect(pattern, mu, sigma.T)
    fi2 = rbf_iter(pattern, mu, sigma)
    #total_error = np.abs(np.dot(fi, weight)-target)**2
    weight = np.linalg.solve(np.dot(fi2.T, fi2), np.dot(fi2.T, target))
    print(weight.shape)
    predict = np.dot(fi2, weight)
    residual = target - predict
    return predict, residual

def delta_rule(patterns, targets, mu):
    sigma = np.ones(mu.shape)
    weights = np.random.normal(mu.shape)
    predicts=[]
    errors=[]
    eta = 0.1

    for pattern, target in zip(patterns, targets):
        fi = rbf_iter(pattern, mu, sigma)
        predict = np.dot(fi.T, weights)
        inst_error = 0.5*(target - predict)
        deltaW = eta*(target-predict)*fi

        weights += deltaW
        predicts.append(predict)
        errors.append(inst_error)
    return predict, errors

pattern = np.arange(0, 2*np.pi, 0.1)
pattern = pattern.reshape((-1,1))
target_sin = np.sin(2*pattern)
target_square = np.sign(target_sin)
mu = np.asarray([np.pi*n/4 for n in range(1,20)])
print(mu)

predict, residual = rbf_learning(pattern, target_sin, mu)

fig1 = plt.figure()
plt.plot(pattern, target_sin)
plt.plot(pattern, predict)

fig2 = plt.figure()
plt.plot(pattern, residual)
plt.show()
