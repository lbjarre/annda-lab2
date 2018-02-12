import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

def generateData(noise=0,offset=0.05):

    t = np.arange(0,2*np.pi,0.1)
    train = np.copy(t)
    valid = t+0.05
    label = np.sin(2*t)
    label = label +np.random.normal(0,0.1,len(train))*noise

    return train, valid, label

def phi_i(x,mu,sigma2=1):
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

def dist_(x, W,r=1.0):
    temp = np.subtract.outer(x,W)
    d = np.multiply(temp,temp).squeeze()
    in_radius = np.abs(d-np.min(d)).T<r
    return d, in_radius

def CLearning(x,W,eta,t=1,rad=1):
    rad = rad**2 #distances are squared
    for i in range(t):
        d = dist_(x,W)
        print(d)
        nbhood = (d.T-np.min(d,1)).T<rad
        delta_W = np.multiply(nbhood,np.subtract.outer(x,W))
        delta_W = eta*np.sum(delta_W,0)
        W += delta_W
    return W

def clearning(x,mu_cl,t=50,eta=0.2):
    for i in range(t):
        x_samp = np.random.choice(x)
        d, in_radius = dist_(x_samp,mu_cl,r=0.25)
        mu_cl += eta*np.multiply(in_radius,x_samp-mu_cl)
    return mu_cl

if __name__ == "__main__":
    x, valid, label = generateData()
    t = 200 #number of epochs
    eta = 0.2 #step size, learning rate
    #mu = np.random.uniform(low=0, high=np.pi*2, size=(10,))
    mu = np.linspace(0,2*np.pi,10)
    f_hat_b, W_b = batch_train(x, label, mu)
    f_hat_seq, W_seq= seq_learn(x,label,mu,t,eta)
    mu_cl = np.copy(mu)
    mu_cl = clearning(x,mu_cl)
    f_hat_b_cl, W_b = batch_train(x, label, mu_cl)
    f_hat_seq_cl, W_seq= seq_learn(x,label,mu_cl,t,eta)

    fig, ax = plt.subplots(2,2)
    plt.subplot(2,2,1)
    plt.plot(x,f_hat_b)
    plt.plot(x,label)
    plt.scatter(mu, np.zeros(mu.shape))
    plt.title('f_hat_batch')


    plt.subplot(2,2,2)
    plt.plot(x,f_hat_seq)
    plt.plot(x,label)
    plt.scatter(mu,np.zeros(mu.shape))
    plt.title('f_hat_seq')

    plt.subplot(2,2,3)
    plt.plot(x,f_hat_seq)
    plt.plot(x,label)
    plt.scatter(mu_cl,np.zeros(mu.shape))
    plt.title('f_hat_seq_cl')

    plt.subplot(2,2,4)
    plt.plot(x,f_hat_seq)
    plt.plot(x,label)
    plt.scatter(mu_cl,np.zeros(mu.shape))
    plt.title('f_hat_b_cl')

    plt.show()
