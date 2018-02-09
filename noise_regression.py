import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

def generateData(offset=0.05):

    t = np.arange(0,2*np.pi,0.1)

    train = t

    valid = t+0.05

    label = np.sin(2*t)
    label = label +np.random.normal(0,0.1,len(train))

    return train, valid, label

def phi_i(x,mu,sigma2=1):
    temp = []
    for i in range(len(mu)):
        phi = np.exp(-((x-mu[i])**2)/(2*sigma2))
        temp.append(phi)
    phi = np.array(temp)
    return phi


def batch_train(x,label,mu,sigma2=1):

    phi_x = phi_i(x, mu)

    A = np.dot(phi_x.T,phi_x)

    b = np.dot(phi_x.T,label)

    W_optimal = np.linalg.solve(A,b)

    f_hat = np.dot(W_optimal,phi_x.T)
    print(f_hat.shape)

    plt.plot(x,f_hat)
    plt.plot(x,label)
    plt.show()

    print (f_hat.shape)

def seq_learn(X,label,mu,t, eta):
    W = np.random.rand(len(mu))*.1
    error = 0
    f_hat = np.zeros((len(X),1))
    for j in range(t):
        for i in range(len(X)):
            phi_x = phi_i(X[i],mu)
            W += eta * error * phi_x
            f_hat[i] = np.dot(W,phi_x)
            error = label[i]-f_hat[i]
    plt.plot(X,f_hat)
    plt.plot(X,label)
    plt.show()
    #delta_W = eta*phi_x
    #error = f-f_hat
    #f_hat = np.dot(W_optimal,phi_x.T)

if __name__ == "__main__":
    x, valid, label = generateData()
    mu = [0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi]
    #batch_train(x, label, mu)
    seq_learn(x,label,mu,5,0.7)
