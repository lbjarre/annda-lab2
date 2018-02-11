import numpy as np
import matplotlib.pyplot as plt

def generate_data(sample_start, step_size):
    pattern = np.arange(sample_start, 2*np.pi, step_size)
    pattern = pattern.reshape((-1,1))

    target_sin = np.sin(2*pattern)
    target_sin += np.random.normal(0, 0.1, target_sin.shape)
    target_square = np.sign(target_sin)

    return pattern, target_sin, target_square

def rbf_iter(X, MU, sigma):
    fi = np.zeros((len(X), len(MU)))
    for i in range(len(X)):
            fi[i,:] = np.exp((-(X[i]-MU)**2)/(2*sigma**2))
    return fi

def rbf_learning(patterns, targets, mu, sigma):
    fi = rbf_iter(patterns, mu, sigma)
    weights = np.linalg.solve(np.dot(fi.T, fi), np.dot(fi.T, targets))
    predicts = np.dot(fi, weights)
    residual = (np.abs(targets-predicts))
    return residual, weights

def delta_rule(patterns, targets, mu, sigma, eta, epochs):
    patterns_targets = np.hstack((patterns, targets))
    weights = np.random.normal(size=(1,len(mu)))
    predicts=np.zeros(targets.shape)
    residuals = []
    for epoch in range(epochs):
        np.random.shuffle(patterns_targets)
        patterns = patterns_targets[:, 0]
        targets = patterns_targets[:, 1]
        for i in range(len(patterns)):
            fi = rbf_iter([patterns[i]], mu, sigma)
            predict = np.dot(fi, weights.T)
            weights += np.squeeze(eta*(targets[i]-predict)*fi)
            predicts[i] = predict
        residuals.append(np.abs(np.mean((targets-predicts))))
    return residuals, weights

if __name__ == "__main__":
    mu = np.asarray([np.pi*n/4 for n in range(0,10)])
    sigma = np.ones(mu.shape)*1
    epochs = 100
    eta = 0.7

    pattern, target_sin, target_square = generate_data(0, 0.1)
    test_pattern, test_target_sin, test_target_square = generate_data(0.05, 0.1)
    test_fi = rbf_iter(test_pattern, mu, sigma)


    """Batch learning"""
    abs_residual_batch, weights_batch = rbf_learning(pattern, target_sin, mu, sigma)
    test_predict_batch = np.dot(test_fi, weights_batch)

    fig1 = plt.figure()
    plt.plot(test_pattern, test_target_sin)
    plt.plot(test_pattern, test_predict_batch)

    fig2 = plt.figure()
    plt.plot(abs_residual_batch)

    """Sequential learning"""
    # abs_residual_seq, weights_seq = delta_rule(pattern, target_sin, mu, sigma, eta, epochs)
    # test_predict_seq = np.dot(test_fi, weights_seq.T)
    #
    #
    # fig3 = plt.figure()
    # plt.plot(test_pattern, test_target_sin, color='r')
    # plt.plot(test_pattern, test_predict_seq, color='b')
    #
    # fig4 = plt.figure()
    # plt.plot(abs_residual_seq)
    plt.show()
