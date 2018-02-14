import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(1234)

def generateData(fun=0, noise=0,offset=0.05):

    t = np.arange(0,2*np.pi,0.1)
    train = np.copy(t)
    valid = t+0.05
    if fun==0:
        label = np.sin(2*t)
    else:
        label = signal.square(2*t)
    label = label + np.random.normal(0,0.3,len(train))*noise

    return train, valid, label

def phi(x,mu,sigma2=0.5):

    temp = []

    for i in range(len(mu)):
        phi = np.exp(-((x-mu[i])**2)/(2*sigma2))
        temp.append(phi)
    phi = np.array(temp)
    return phi

def batch_train(x,label,phi_x,sigma2=1):

    A = np.dot(phi_x.T,phi_x)
    b = np.dot(phi_x.T,label)
    W_optimal = np.linalg.solve(A,b)

    f_hat = np.sign(np.dot(W_optimal,phi_x.T))

    error = np.mean((f_hat-label)**2)

    return f_hat, W_optimal, error

def seq_learn(x,label,mu,t, eta):
    W = np.random.rand(len(mu))*.1
    phi_x = phi(x, mu)
    error = []

    for j in range(t):
        for i in range(len(x)):
            f_hat = np.dot(W, phi_x[:,i])
            e = label[i]-f_hat
            delta_W = eta*e*phi_x[:,i]

            W += delta_W

        f_temp = np.dot(W,phi_x)
        tot_err = (f_temp-label).T*(f_temp-label)
        error.append(np.mean(tot_err))

    f_hat = np.dot(W,phi_x)

    return f_hat, W, error

def dist_(x, W, r=.5):
    temp = np.subtract.outer(x,W)
    d = np.multiply(temp,temp).squeeze()
    in_radius = np.abs(d-np.min(d)).T<r
    return d, in_radius

def clearning(x,mu_cl,t=50,eta=0.2):
    for i in range(t):
        x_samp = np.random.choice(x)
        d, in_radius = dist_(x_samp,mu_cl,r=1)
        mu_cl += eta*np.multiply(in_radius,x_samp-mu_cl)
    return mu_cl

def winner(x,mu,fac=5):
    winners = np.argmin(dist_(x,mu)[0],1)
    wins = np.zeros(mu.shape)
    for w in winners:
        wins[w] += 1

    wins = wins*fac

    return wins

def batch_plots(sigma2=1):
    x, valid, label = generateData(fun=1, noise=0)

    no_of_nodes = np.arange(1,16,1)
    errors = []

    for i in no_of_nodes:
        #mu = np.random.uniform(low=0, high=np.pi*2, size=(i,))
        mu = np.linspace(0,2*np.pi,i)
        phi_x = phi(x,mu,sigma2).T
        f_hat_b, W_b, error = batch_train(x, label, phi_x)
        errors.append(error)

    print(errors)

    fig = plt.figure()

    #plt.plot(no_of_nodes, errors)
    tru, = plt.plot(x,f_hat_b, c="b", label="Estimated")
    est, = plt.plot(x,label, '--r', label="True")

    nodes = plt.scatter(mu, np.zeros(mu.shape), label="Nodes")
    plt.legend(handles=[est, tru, nodes])
    plt.title('Square(2x): Sigma = ' + str(sigma2) + ' - 15 nodes')

    fig.savefig('report/plots/batch/best_square_cheat')

    plt.show()

def assignment1():
    x, valid, label = generateData()
    t = 100 #number of epochs
    eta = 0.2 #step size, learning rate
    mu = np.random.uniform(low=0, high=np.pi*2, size=(7,))
    #mu = np.linspace(0,2*np.pi,6)
    f_hat_b, W_b, erb = batch_train(x, label, mu)
    winners = winner(x,mu)

    f_hat_seq, W_seq, error_s = seq_learn(x,label,mu,t,eta)
    mu_cl = np.copy(mu)
    mu_cl = clearning(x,mu_cl)
    winners_cl = winner(x,mu_cl)
    f_hat_b_cl, W_b_cl, erb_cl = batch_train(x, label, mu_cl)
    f_hat_seq_cl, W_seq_cl, error_s_cl = seq_learn(x,label,mu_cl,t,eta)

    phi_x = phi(valid,mu)
    f_hat_b_test = np.dot(W_b,phi_x)
    f_hat_seq_test = np.dot(W_seq,phi_x)
    phi_x_cl = phi(valid,mu_cl)
    f_hat_b_test_cl = np.dot(W_b_cl,phi_x_cl)
    f_hat_seq_test_cl = np.dot(W_seq_cl,phi_x_cl)

    plt.plot(error_s)
    #plt.show()

    fig, ax = plt.subplots(2,2)
    plt.subplot(2,2,1)
    plt.plot(x,f_hat_b,label="Predicted")
    plt.plot(x,label,label="True")
    plt.scatter(mu, np.zeros(mu.shape),s=winners)
    plt.title('f_hat_batch')

    plt.subplot(2,2,2)
    plt.plot(x,f_hat_seq,label="Predicted")
    plt.plot(x,label, label="True")
    plt.scatter(mu,np.zeros(mu.shape),s=winners)
    plt.title('f_hat_seq')

    plt.subplot(2,2,3)
    plt.plot(x,f_hat_b_cl,label="Predicted")
    plt.plot(x,label, label="True")
    plt.scatter(mu_cl,np.zeros(mu.shape),s=winners_cl)
    plt.title('f_hat_b_cl')

    plt.subplot(2,2,4)
    plt.plot(x,f_hat_seq_cl,label="Predicted")
    plt.plot(x,label, label="True")
    plt.scatter(mu_cl,np.zeros(mu.shape),s=winners_cl)
    plt.title('f_hat_seq_cl')

    #plt.show()

    fig, ax = plt.subplots(2,2)
    plt.subplot(2,2,1)
    plt.plot(valid,f_hat_b_test,label="Predicted")
    plt.plot(valid,label, label="True")
    plt.scatter(mu, np.zeros(mu.shape),s=winners)
    plt.title('f_hat_batch_test')

    plt.subplot(2,2,2)
    plt.plot(valid,f_hat_seq_test,label="Predicted")
    plt.plot(valid,label, label="True")
    plt.scatter(mu,np.zeros(mu.shape),s=winners)
    plt.title('f_hat_seq_test')

    plt.subplot(2,2,3)
    plt.plot(valid,f_hat_b_test_cl,label="Predicted")
    plt.plot(valid,label, label="True")
    plt.scatter(mu_cl,np.zeros(mu.shape),s=winners_cl)
    plt.title('f_hat_b_batch_test')

    plt.subplot(2,2,4)
    plt.plot(valid,f_hat_seq_test_cl,label="Predicted")
    plt.plot(valid,label, label="True")
    plt.scatter(mu_cl,np.zeros(mu.shape),s=winners_cl)
    plt.title('f_hat_b_cl_test')

    #plt.show()


if __name__ == "__main__":
    #assignment1()
    batch_plots()
