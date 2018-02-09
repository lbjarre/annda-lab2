import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    pattern = np.arange(0, 2*np.pi, 0.1)
    pattern = pattern.reshape((-1,1))

    target_sin = np.sin(2*pattern)
    target_sin += np.random.normal(0, 0.1, target_sin.shape)
    target_square = np.sign(target_sin)

    return pattern, target_sin, target_square

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
    fi2 = rbf_iter(pattern, mu, sigma)
    weight = np.linalg.solve(np.dot(fi2.T, fi2), np.dot(fi2.T, target))
    predict = np.dot(fi2, weight)
    residual = target - predict

    return predict, np.abs(residual)

def delta_rule(patterns, targets, mu, eta, epochs):
    sigma = np.ones(mu.shape)
    weights = np.random.normal(size=(1,len(mu)))
    predicts=np.zeros(targets.shape)
    residuals = []
    for epoch in range(epochs):
        for i in range(len(patterns)):
            fi = rbf_iter(patterns[i], mu, sigma)
            predict = np.dot(fi, weights.T)
            weights += np.squeeze(eta*(targets[i]-predict)*fi)
            predicts[i] = predict
        residuals.append(np.abs(np.mean((targets-predicts))))
    return predicts, residuals

if __name__ == "__main__":
    pattern, target_sin, target_square = generate_data()
    mu = np.asarray([np.pi*n/4 for n in range(0,50)])
    epochs = 100
    eta = 0.5
    predict_batch, residual = rbf_learning(pattern, target_sin, mu)
    predict_seq, abs_residual = delta_rule(pattern, target_sin, mu, eta, epochs)

    # fig1 = plt.figure()
    # plt.plot(pattern, target_sin)
    # plt.plot(pattern, predict_batch)
    #
    # fig2 = plt.figure()
    # plt.plot(pattern, residual)
    # plt.show()

    fig3 = plt.figure()
    plt.plot(pattern,target_sin, color='r')
    plt.plot(pattern, predict_seq, color='b')
    fig4 = plt.figure()

    plt.plot(abs_residual)
    plt.show()
