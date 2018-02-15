import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

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

def initMu(x, mu, t=20, eta=0.2, n=10,rad=1):
    #randomly select input points
    temp = np.zeros((n,len(mu)))

    x_samp = np.zeros((n,x.shape[1]))

    ind = np.arange(0,x.shape[0])

    for j in range(t):
        #sample x
        for i in range(n):
            index = np.random.choice(ind)
            x_samp[i,:] = x[index,:]

        for i in range(n):
            #calculate closest node
            d = np.sum((x_samp[i,:]-mu)**2,1)
            nbh = np.abs((d-np.min(d)))
            nbh = nbh<=rad
            #closest nodes are updated
            mu += x_samp[i,:] - np.multiply(nbh.reshape(-1,1),mu)

    return mu

def phi(x,mu,sigma2=0.5):
    temp = []
    for i in range(len(mu)):
        phi = np.exp(np.sum(-(x-mu[i])**2/(2*sigma2),1))
        temp.append(phi)
    phi = np.array(temp).T
    return phi

def batch_train(x,label,mu,sigma2=1):

    phi_x = phi(x, mu)

    A = np.dot(phi_x.T,phi_x)
    b = np.dot(phi_x.T,label)
    W = np.linalg.solve(A,b)

    f_hat = np.dot(phi_x,W)

    return f_hat, W

def seq_learn(x,label,mu,t, eta):
    W = np.random.rand(mu.shape[0],mu.shape[1])*.1
    phi_x = phi(x, mu)

    for j in range(t):
        for i in range(len(x)):
            f_hat = np.dot(W.T, phi_x[i,:])
            e = label[i]-f_hat
            delta_W = eta*np.outer(e,phi_x[i,:]).T

            W += delta_W

    f = np.dot(phi_x,W)

    return f, W

def assignment1_ballist():
    train_data, train_labels, test_data, test_labels = loadData()

    no_of_nodes = 20

    mu = []
    ind = np.arange(0,train_data.shape[0])
    for i in range(no_of_nodes):
        index = np.random.choice(ind)
        mu.append(train_data[index,:])
    mu = np.array(mu)

    t = 150
    eta = 0.2

    f_hat_b, W_b = batch_train(train_data, train_labels, mu)
    f_hat_s, W_s = seq_learn(train_data, train_labels, mu, t, eta)

    fig = plt.figure()
    tr = plt.scatter(train_labels[:,0],train_labels[:,1], c='g', label="true")
    f_h_s = plt.scatter(f_hat_s[:,0],f_hat_s[:,1], c='b', label="Seql")
    f_h_b = plt.scatter(f_hat_b[:,0], f_hat_b[:,1], c='r', label="Batch")

    plt.title('Batch and sequential: 150 epochs, eta: 0.2')
    plt.legend(handles=[f_h_s, f_h_b, tr])
    fig.savefig('report/plots/2d/first_basic_both_no_CL')
    plt.show()


if __name__ == '__main__':
    assignment1_ballist()
