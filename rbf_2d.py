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

def dist_(x, W, r=.5):

    temp = []
    for w in W:
        d = x-w
        d = d[0]**2 + d[1]**2
        temp.append(d)

    d = temp
    in_radius = (d-np.min(d))<r

    """
    temp = np.subtract.outer(x,W)
    d = temp**2
    in_radius = np.abs(d-np.min(d)).T<r
    """
    return d, in_radius

def clearning(x,mu_cl,t=50,eta=0.2,r=.5):
    a = np.arange(x.shape[0])
    for i in range(t):
        x_samp = x[np.random.choice(a),:]
        d, in_radius = dist_(x_samp, mu_cl,r=r)
        mu_cl[np.argmin(d),:] += eta*(x_samp-mu_cl[np.argmin(d),:])
    return mu_cl

def assignment1_ballist():
    train_data, train_labels, test_data, test_labels = loadData()

    no_of_nodes = 5

    mu = []
    ind = np.arange(0,train_data.shape[0])
    for i in range(no_of_nodes):
        index = np.random.choice(ind)
        mu.append(train_data[index,:])

    mu = np.array(mu)
    mu_cp = np.copy(mu)
    mu_cl = clearning(train_data, mu_cp)

    t = 150
    eta = 0.2

    f_hat_b, W_b = batch_train(train_data, train_labels, mu)
    f_hat_s, W_s = seq_learn(train_data, train_labels, mu, t, eta)

    f_hat_b_cl, W_b_cl = batch_train(train_data, train_labels, mu)
    f_hat_s_cl, W_s_cl = seq_learn(train_data, train_labels, mu, t, eta)

    #output space
    fig = plt.figure()
    tr = plt.scatter(train_labels[:,0],train_labels[:,1], c='g', label="true")
    f_h_s = plt.scatter(f_hat_s[:,0],f_hat_s[:,1], c='b', label="Seql")
    f_h_b = plt.scatter(f_hat_b[:,0], f_hat_b[:,1], c='r', label="Batch")
    f_h_s_cl = plt.scatter(f_hat_s[:,0],f_hat_s[:,1], c='b', label="Seql cl")
    f_h_b_cl = plt.scatter(f_hat_b[:,0], f_hat_b[:,1], c='r', label="Batch cl")
    plt.title('Batch and sequential: 150 epochs, eta: 0.2')
    plt.legend(handles=[f_h_s, f_h_b, f_h_b_cl, f_h_s_cl, tr])
    #fig.savefig('report/plots/2d/first_basic_both_no_CL_output')

    plt.show()

    #input space
    fig2 = plt.figure()
    train_data = plt.scatter(train_data[:,0],train_data[:,1], c='g', label="data")
    nds = plt.scatter(mu[:,0],mu[:,1], c='b', label="Nodes")
    nds_cl = plt.scatter(mu_cl[:,0],mu_cl[:,1], c='r', label="Nodes CL")
    plt.legend(handles=[train_data, nds, nds_cl])
    plt.title("Training data and nodes")
    #fig2.savefig('report/plots/2d/input_basic_both_cl_INPUTAOSHAS')

    plt.show()


if __name__ == '__main__':
    assignment1_ballist()
