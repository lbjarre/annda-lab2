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
        valid_label = np.sin(2*valid)
    else:
        label = signal.square(2*t)
        valid_label = signal.square(2*valid)

    noi =np.random.normal(0,0.1,len(train))*noise
    label = label + noi
    valid_label = valid_label + noi

    return train, valid, label, valid_label

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

    f_hat = np.dot(W_optimal,phi_x.T)
    #f_hat = np.sign(f_hat)

    error = np.mean((f_hat-label)**2)

    return f_hat, W_optimal, error

def seq_learn(x, label, phi_x, t, eta):
    W = np.random.rand(phi_x.shape[1])*.1
    error = []
    order = np.arange(len(x))

    phi_x_start = np.copy(phi_x)

    for j in range(t):
        np.random.shuffle(order)
        x = x[order].squeeze()
        label = label[order].squeeze()
        phi_x = phi_x[order,:].squeeze()

        for i in range(x.shape[0]):
            f_hat = np.dot(W, phi_x[i,:].T)
            e = label[i]-f_hat
            delta_W = eta*e*phi_x[i,:]
            W += delta_W

        f_temp = np.dot(W,phi_x.T)
        tot_err = (f_temp-label)**2


    f_hat = np.dot(W,phi_x_start.T)

    error = np.mean((f_hat-label)**2)

    return f_hat, W, error

def dist_(x, W, r=.5):
    temp = np.subtract.outer(x,W)
    d = temp**2
    in_radius = np.abs(d-np.min(d)).T<r
    return d, in_radius

def clearning(x,mu_cl,t=50,eta=0.2,r=.5):
    for i in range(t):
        x_samp = np.random.choice(x)
        d, in_radius = dist_(x_samp,mu_cl,r=r)
        mu_cl += eta*np.multiply(in_radius,x_samp-mu_cl)
    return mu_cl

def winner(x,mu,fac=10):
    winners = np.argmin(dist_(x,mu)[0],1)
    wins = np.zeros(mu.shape)
    for w in winners:
        wins[w] += 1

    wins = wins*fac

    return wins

def batch_plots(sigma2=.02):
    x, valid, label, valid_label = generateData(fun=0, noise=1)

    no_of_nodes = np.arange(1,16,1)
    errors = []
    test_error = []

    for i in no_of_nodes:
        #mu = np.random.uniform(low=0, high=np.pi*2, size=(i,))
        mu = np.linspace(0,2*np.pi,i)
        phi_x = phi(x,mu,sigma2).T
        f_hat_b, W_b, error = batch_train(x, label, phi_x)
        phi_test = phi(valid, mu, sigma2)
        f_test = np.dot(W_b, phi_test)
        #f_test = np.sign(f_test)
        errors.append(error)
        test_error.append(np.mean((f_test-valid_label)**2))

    phi_test = phi(valid, mu, sigma2)

    f_test = np.dot(W_b, phi_test)
    #f_test = np.sign(f_test)
    print(test_error)
    print(np.min(test_error))

    fig = plt.figure()

    train, = plt.plot(x,f_hat_b, c="b", label="Training")
    true, = plt.plot(x,label, '--r', label="True")
    test, = plt.plot(valid, f_test, label="Test")
    #plt.scatter(valid, f_test)

    nodes = plt.scatter(mu, np.zeros(mu.shape), label="Nodes")
    plt.legend(handles=[train, test, true, nodes, ])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Batch: sin(2x), with noise: Sigma = ' + str(sigma2) + " Nodes: " + str(i))

    #fig.savefig('report/plots/noise/batch_sin2x_sigma_002.png')

    plt.show()

    fig2 = plt.figure()

    tra, = plt.plot(no_of_nodes, errors, label="training")
    test_er, = plt.plot(no_of_nodes, test_error, label="test")
    plt.title('sin(2x), with noise: Sigma = ' + str(sigma2))
    plt.xlabel('# nodes')
    plt.ylabel('error')
    plt.legend(handles=[tra, test_er])

    #fig2.savefig('report/plots/noise/batch_sin2x_error_sigma_002.png')

    plt.show()

def seq_plots(sigma2=1):
    x, valid, label, valid_label = generateData(0, noise=1)
    t = 100 #number of epochs
    eta = 0.2 #step size, learning rate

    #no_of_epochs = np.arange(1,101,1)
    no_of_nodes = np.arange(2,16,1)
    tot_errors = []
    test_error = []
    #i = 15

    for i in no_of_nodes:
        #mu = np.random.uniform(low=0, high=np.pi*2, size=(i,))
        mu = np.linspace(0,2*np.pi,i)
        phi_x = phi(x,mu,sigma2).T
        f_hat, W, error = seq_learn(x, label, phi_x, t, eta)
        f_hat = np.dot(W, phi_x.T)
        error_outside = np.mean((f_hat-label)**2)
        #print(np.mean(error))
        tot_errors.append(error_outside)

        phi_test = phi(valid, mu, sigma2)
        f_test = np.dot(W, phi_test)
        test_error.append(np.mean((f_test-valid_label)**2))

    fig = plt.figure()

    plt.title('sin(2x), with noise: Sigma = ' + str(sigma2) + ' Nodes: ' + str(i) + ' eta: ' + str(eta))
    train_plot, = plt.plot(no_of_nodes, tot_errors, label="Training Err")
    test_plot, = plt.plot(no_of_nodes, test_error, label="Test Err")
    plt.xlabel('# nodes')
    plt.ylabel('error')
    plt.legend(handles=[train_plot, test_plot])

    #seq_sin2x_100ep_sigma1_error
    #fig.savefig('report/plots/noise/seq_sin2x_'+str(t)+'ep_sigma1')
    fig.savefig('report/plots/noise/seq_sin2x_'+str(t)+'ep_sigma1_error')

    plt.show()

    fig2 = plt.figure()
    est, = plt.plot(x, f_hat, c="b", label="Estimated")
    tru, = plt.plot(x, label, '--r', label="True")
    test, = plt.plot(valid, f_test, label="Test")

    nodes = plt.scatter(mu, np.zeros(mu.shape), label="Nodes")
    plt.legend(handles=[est,test, tru, nodes])
    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.title('sin(2x), with noise: Sigma = ' + str(sigma2) + ' Epochs: ' + str(t) + ' eta: ' + str(eta) + ' Nodes: ' + str(i))

    #fig2.savefig('report/plots/noise/seq_sin2x_'+str(t)+'ep_sigma1')

    #plt.show()

def CL_plots(sigma2=0.5):
    x, valid, label, valid_label = generateData(0, noise=1)
    epochs = 100#number of epochs
    eta = 0.2 #step size, learning rate

    no_of_nodes = np.arange(2,16,1)
    tot_errors_cl = []
    tot_errors = []

    for i in no_of_nodes:
        mu = np.linspace(0,2*np.pi,i)
        mu_copy = np.copy(mu)
        mu_cl = clearning(x, mu_copy, t=100,r=0.01)
        phi_x_cl = phi(x,mu_cl,sigma2).T
        f_hat_cl, W_cl, error_cl = seq_learn(x, label, phi_x_cl, epochs, eta)
        #f_hat_cl, W, error_cl = batch_train(x, label, phi_x_cl, sigma2)

        phi_x = phi(x,mu,sigma2).T
        f_hat, W, error = seq_learn(x, label, phi_x, epochs, eta)

        phi_test = phi(valid, mu, sigma2)

        f_test = np.dot(W, phi_test)
        f_test_cl = np.dot(W_cl, phi_test)

        #f_hat, W, error = batch_train(x, label, phi_x, sigma2)
        tot_errors_cl.append(np.mean((f_test_cl-valid_label)**2))
        tot_errors.append(np.mean((f_test-valid_label)**2))

    winners_cl = winner(valid,mu_cl)

    fig = plt.figure()

    tru, = plt.plot(x, f_hat_cl, c="b", label="Estimated - CL")
    est, = plt.plot(x, label, '--r', label="True")
    no_cl, = plt.plot(x, f_hat, 'g', label="No CL")

    nodes = plt.scatter(mu_cl, np.zeros(mu.shape), label="Nodes CL", s=winners_cl)
    nodes_no = plt.scatter(mu, np.zeros(mu.shape), label="Nodes no CL")
    plt.legend(handles=[est, no_cl, tru, nodes, nodes_no])
    plt.xlabel('x')
    plt.ylabel('f(x)')

    #fig.savefig('report/plots/cl/')

    plt.show()

    plt.title('Sin(2x) seq, Sigma = ' + str(sigma2) + ' Epochs: ' + str(epochs) + ' eta: ' + str(eta) + ' Nodes: ' + str(i))

    fig2 = plt.figure()
    plt.title('Sin(2x), with noise: Sigma = ' + str(sigma2))
    CL, = plt.plot(no_of_nodes, tot_errors_cl, label="CL")
    NoCL, = plt.plot(no_of_nodes, tot_errors, label="No CL")
    plt.xlabel('# nodes')
    plt.ylabel('error')
    plt.legend(handles=[CL, NoCL])

    #fig2.savefig('report/plots/cl/')

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
    #batch_plots()
    #seq_plots()
    CL_plots()
