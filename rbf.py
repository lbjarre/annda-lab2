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
    phi = np.zeros((len(X), len(MU)))
    for i in range(len(X)):
            phi[i,:] = np.exp((-(X[i]-MU)**2)/(2*sigma**2))
    return phi

def rbf_learning(patterns, targets, mu, sigma):
    phi = rbf_iter(patterns, mu, sigma)
    weights = np.linalg.solve(np.dot(phi.T, phi), np.dot(phi.T, targets))
    predicts = np.dot(phi, weights)
    abs_residual = np.abs(np.mean(targets-predicts))
    return abs_residual, weights

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
            phi = rbf_iter([patterns[i]], mu, sigma)
            #phi = CL_algorithm(patterns, phi, eta)
            predict = np.dot(phi, weights.T)
            weights += np.squeeze(eta*(targets[i]-predict)*phi)
            predicts[i] = predict
        residuals.append(np.abs(np.mean(targets-predicts)))
    return residuals, weights

def CL_algorithm(patterns, phi, eta):
    pattern = np.random.choice(patterns)
    dist = np.linalg.norm(pattern-phi, axis=0)
    min_arg = np.argmin(dist)
    phi[0, min_arg] = eta*(pattern-phi[0, min_arg])
    print(phi.shape)
    return phi

if __name__ == "__main__":
    mu = np.asarray([np.pi*n/4 for n in range(0,30)])
    sigma = np.ones(mu.shape)*1
    epochs = 1000
    eta = 0.7

    pattern, target_sin, target_square = generate_data(0, 0.1)
    test_pattern, test_target_sin, test_target_square = generate_data(0.05, 0.1)
    test_phi = rbf_iter(test_pattern, mu, sigma)


    """Batch learning"""
    # abs_residual_batch, weights_batch = rbf_learning(pattern, target_sin, mu, sigma)
    # test_predict_batch = np.dot(test_phi, weights_batch)
    #
    # fig1 = plt.figure()
    # plt.plot(test_pattern, test_target_sin)
    # plt.plot(test_pattern, test_predict_batch)
    # print("Average absolute residual: {}".format(abs_residual_batch))

    """Sequential learning"""
    abs_residual_seq, weights_seq = delta_rule(pattern, target_sin, mu, sigma, eta, epochs)
    test_predict_seq = np.dot(test_phi, weights_seq.T)


    #fig3 = plt.figure()
    plt.plot(test_pattern, test_target_sin, color='r')
    plt.plot(test_pattern, test_predict_seq, color='b')

    fig4 = plt.figure()
    plt.plot(abs_residual_seq)
    plt.show()
