import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(1234)

def generateData(noise=1,offset=0.05):

    t = np.arange(0,2*np.pi,0.1)
    train = np.copy(t)
    valid = t+0.05
    label = np.sin(2*t)
    label = label + np.random.normal(0,0.3,len(train))

    return train, valid, label

def phi_i(x,mu,sigma2=0.5):
    temp = []
    for i in range(len(mu)):
        phi = np.exp(-((x-mu[i])**2)/(2*sigma2))
        temp.append(phi)
    phi = np.array(temp)
    return phi


def batch_train(x,label,mu,sigma2=1):

    phi_x = phi_i(x, mu).T

    A = np.dot(phi_x.T,phi_x)
    b = np.dot(phi_x.T,label)
    W_optimal = np.linalg.solve(A,b)

    f_hat = np.dot(W_optimal,phi_x.T)


    return f_hat, W_optimal

def seq_learn(x,label,mu,t, eta):
    W = np.random.rand(len(mu))*.1
    phi_x = phi_i(x, mu)

    for j in range(t):
        for i in range(len(x)):
            f_hat = np.dot(W, phi_x[:,i])
            e = label[i]-f_hat
            delta_W = eta*e*phi_x[:,i]

            W += delta_W

    f_hat = np.dot(W,phi_x)

    return f_hat, W

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

def loadData():
    data = []
    with open('data_lab2/ballist.dat', 'r') as f:
        next = f.readline()
        while next != "":
            list = next.replace('\t',' ').replace('\n', '').split(' ')
            list = [float(i) for i in list]
            data.append(list)
            next = f.readline()

    train = np.copy(data)
    train_data = train[:,0:2]
    train_labels = train[:,2:4]

    data = []
    with open('data_lab2/balltest.dat', 'r') as f:
        next = f.readline()
        while next != "":
            list = next.replace('\t',' ').replace('\n', '').split(' ')
            list = [float(i) for i in list]
            data.append(list)
            next = f.readline()

    test = np.copy(data)
    test_data = test[:,0:2]
    test_labels = test[:,2:4]

    return train_data, train_labels, test_data, test_labels

def initMu(n, d):
    for i in range(t):
        x_samp = np.random.choice(x)
        d, in_radius = dist_(x_samp,mu_cl,r=1)
        mu_cl += eta*np.multiply(in_radius,x_samp-mu_cl)
    return mu_cl

def assignment1():
    x, valid, label = generateData()
    t = 100 #number of epochs
    eta = 0.2 #step size, learning rate
    mu = np.random.uniform(low=0, high=np.pi*2, size=(6,))
    #mu = np.linspace(0,2*np.pi,6)
    f_hat_b, W_b = batch_train(x, label, mu)
    winners = winner(x,mu)

    f_hat_seq, W_seq = seq_learn(x,label,mu,t,eta)
    mu_cl = np.copy(mu)
    mu_cl = clearning(x,mu_cl)
    winners_cl = winner(x,mu_cl)
    f_hat_b_cl, W_b_cl = batch_train(x, label, mu_cl)
    f_hat_seq_cl, W_seq_cl = seq_learn(x,label,mu_cl,t,eta)

    phi_x = phi_i(valid,mu)
    f_hat_b_test = np.dot(W_b,phi_x)
    f_hat_seq_test = np.dot(W_seq,phi_x)
    phi_x_cl = phi_i(valid,mu_cl)
    f_hat_b_test_cl = np.dot(W_b_cl,phi_x_cl)
    f_hat_seq_test_cl = np.dot(W_seq_cl,phi_x_cl)

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

    plt.show()

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


    plt.show()


def assignment1_ballist():
    train_data, train_labels, test_data, test_labels = loadData()

    no_of_nodes = 6

    mu = []
    ind = np.arange(0,train_data.shape[0])
    for i in range(no_of_nodes):
        index = np.random.choice(ind)
        mu.append(train_data[index,:])
    mu = np.array(mu)

    mu_cl = clearning(train_data,mu)


if __name__ == "__main__":
    assignment1()
    #assignment1_ballist()
