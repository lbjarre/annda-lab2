import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def generate_data(sample_start, step_size):
    pattern = np.arange(sample_start, 2*np.pi, step_size)
    target_sin = np.sin(2*pattern)
    target_sin += np.random.normal(0, 0.1, target_sin.shape)
    target_square = np.sign(target_sin)
    return pattern, target_sin, target_square

def generate_dict(pos, dict_names, data):
    dict_votes = {}
    for j in dict_names.keys():
        dict_votes[j] = np.zeros((10,10))
    for i in range(pos.shape[0]):
        dict_i = data[i]
        dict_votes[dict_i][pos[i,:][0], pos[i,:][1]] +=1
    return dict_votes

def rbf_iter(x, mu, sigma2=1):
    temp = []
    for i in range(len(mu)):
        phi = np.exp(-((x-mu[i])**2)/(2*sigma2))
        temp.append(phi)
    phi = np.array(temp)
    return phi

def rbf_learning(patterns, targets, mu):
    phi = rbf_iter(patterns, mu).T
    weights = np.linalg.solve(np.dot(phi.T, phi), np.dot(phi.T, targets))
    predicts = np.dot(weights, phi.T)
    abs_residual = np.abs(np.mean(targets-predicts))
    return abs_residual, weights, predicts

def delta_rule(patterns, targets, mu, eta, epochs):
    patterns_targets = np.hstack((patterns, targets))
    weights = np.random.rand(len(mu))*.1
    phi = rbf_iter(patterns, mu)
    residuals = []
    for epoch in range(epochs):
        # np.random.shuffle(patterns_targets)
        # patterns = patterns_targets[:, 0]
        # targets = patterns_targets[:, 1]
        for i in range(len(patterns)):
            predict = np.dot(weights, phi[:,i])
            error = targets[i] - predict
            delta_W = eta*error*phi[:,i]
            weights += delta_W
    return residuals, weights


def animal_order(props, weights, epochs):
    nbh=50
    pos = []
    eta=0.2
    nAnimals = 32
    # print(props.shape)
    # print(weights.shape)
    for epoch in range(epochs):
        nbh-= 2.5
        for i in range(nAnimals):
            distances = []
            curr_city = props[i,:]
            # print(curr_city.shape)
            # print(weights.shape)
            for j in range(weights.shape[0]):
                A = curr_city-weights[j,:]
                distance = np.dot((curr_city-weights[j,:]).T, (curr_city-weights[j,:]))
                distances.append(distance)
            distances = np.array(distances)
            idx_win = np.argmin(distances)
            # print("Lower bound {}, upper bound {} ".format(idx_win-nbh, idx_win+nbh))
            for t in range(int(idx_win-nbh), int(idx_win+nbh)):
                if t >= 100:
                    t = -(t-100)
                if t<0:
                    t = 100+t
                weights[t,:] = weights[t,:] + eta*(curr_city-weights[t,:])
                # if(t>=0 and t<100):
                #     weights[t,:] = weights[t,:] + eta*(curr_city-weights[t,:])

    for k in range(32):
        distances=[]
        curr_city = props[k,:]
        for l in range(weights.shape[0]):
            distance = np.dot((curr_city-weights[l,:]).T, (curr_city-weights[l,:]))
            distances.append(distance)
        distances = np.array(distances)
        idx_win = np.argmin(distances)
        idx_win = np.argmin(distances)
        pos.append(idx_win)
    return pos

def cyclic_tour(cities, weights, epochs):
    eta = 0.2
    nbh = 2
    n_cities = cities.shape[0]
    pos = []
    for epoch in range(epochs):
        if epoch == epochs/2 or epoch == epochs-5:
            nbh-=1

        for i in range(n_cities):
            distances = []
            curr_city = cities[i,:]
            for j in range(weights.shape[0]):
                distance = np.dot((curr_city-weights[j,:]).T, (curr_city-weights[j,:]))
                distances.append(distance)
            distances = np.array(distances)
            idx_win = np.argmin(distances)
            # print("Lower bound {}, upper bound {} ".format(idx_win-nbh, idx_win+nbh))
            for t in range(int(idx_win-nbh), int(idx_win+nbh)):
                if t >= weights.shape[0]:
                    t = -(t-weights.shape[0])
                if t<0:
                    t = weights.shape[0]+t
                weights[t,:] = weights[t,:] + eta*(curr_city-weights[t,:])

    for k in range(n_cities):
        distances=[]
        curr_city = cities[k,:]
        for l in range(weights.shape[0]):
            distance = np.dot((curr_city-weights[l,:]).T, (curr_city-weights[l,:]))
            distances.append(distance)
        distances = np.array(distances)
        idx_win = np.argmin(distances)
        idx_win = np.argmin(distances)
        pos.append(idx_win)
    return pos

def votes_of_mps(data, weights, epochs):
    n_nodes = weights.shape[0]
    n_mps = data.shape[0]
    eta = 0.1
    nbh_factor = 3

    for epoch in range(epochs):
        nbh = nbh_factor*(np.exp(-epoch/100))
        nbh = int(nbh)
        for i in range(n_mps):
            curr_votes = data[i,:]
            distances = np.linalg.norm(curr_votes - weights, axis=2)
            idx_win = np.unravel_index(np.argmin(distances), distances.shape)
            for t in range(int(idx_win[0]-nbh), int(idx_win[0]+nbh)):
                if t>=0 and t<n_nodes:
                    for s in range(int(idx_win[1]-nbh), int(idx_win[1]+nbh)):
                        if s>=0 and s<n_nodes:
                            weights[t,s,:] = weights[t,s,:] + eta*(curr_votes-weights[t,s,:])
    pos = []
    for j in range(n_mps):
        curr_votes = data[j,:]
        distances = np.linalg.norm(curr_votes - weights, axis=2)
        idx_win = np.unravel_index(np.argmin(distances), distances.shape)
        pos.append(idx_win)
    return pos

if __name__ == "__main__":
    mu = np.linspace(0,2*np.pi,20)
    #sigma = np.ones(mu.shape)*1
    epochs = 300
    eta = 0.2

    pattern, target_sin, target_square = generate_data(0, 0.1)
    test_pattern, test_target_sin, test_target_square = generate_data(0.05, 0.1)

    """Batch learning"""
    # abs_residual, weights, predicts = rbf_learning(pattern, target_sin, mu)
    # test_phi = rbf_iter(test_pattern, mu)
    # test_predict = np.dot(weights, test_phi)
    # plt.plot(pattern, target_sin)
    # plt.plot(pattern, predicts)
    # plt.figure()
    # plt.plot(test_pattern, test_predict)

    """Sequential"""
    # abs_residuals, weights_seq = delta_rule(pattern, target_sin, mu, eta, epochs)
    # test_phi = rbf_iter(test_pattern, mu)
    # test_predict = np.dot(weights_seq, test_phi)
    # plt.plot(test_pattern, test_target_sin)
    # plt.plot(test_pattern, test_predict)
    # plt.show()

    # """SOM-algorithm animals"""
    # with open("data_lab2/animals.dat") as f:
    #     lines = f.readlines()
    #     props = [line.split(',') for line in lines]
    #     props = [int(i) for i in props[0]]
    #     props = np.array(props)
    #     props = props.reshape((32, 84))
    #
    # with open("data_lab2/animalnames.txt") as f:
    #     lines = f.readlines()
    #     names = [line.strip('\t\n') for line in lines]
    #     names = np.array(names)
    #
    # epochs = 20
    # weights = np.random.normal(size=(100,84))
    # pos = animal_order(props, weights, epochs)
    # sorted_pos = np.sort(pos)
    #
    # sorted_animals = [x for _,x in sorted(zip(pos, names))]
    # print(sorted_animals)


    # """SOM-algorithm cities"""
    # with open("modded_data/cities.dat") as f:
    #     lines = f.readlines()
    #     data = [line.split(',') for line in lines]
    #     data = [float(i) for i in data[0]]
    #     data = np.array(data)
    #     data = data.reshape((10, 2))
    #
    # epochs = 20
    # weights = np.random.normal(size=(10,2))
    # pos = cyclic_tour(data, weights, epochs)
    # sorted_cities = [*sorted(zip(data, pos), key= lambda x: x[1])]
    # sorted_cities = np.array([x[0] for x in sorted_cities])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.annotate('Start', xy=sorted_cities[0,:])
    # ax.annotate('End', xy=sorted_cities[-1,:])
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.plot(sorted_cities[:,0], sorted_cities[:,1])
    # plt.show()

    """SOM-algorithm votes of MPs"""
    with open("data_lab2/votes.dat") as f:
        lines = f.readlines()
        votes = [line.split(',') for line in lines]
        votes = [float(i) for i in votes[0]]
        votes = np.array(votes)
        votes = votes.reshape((349, 31))

    with open("modded_data/mpparty.dat") as f:
        lines = f.readlines()
        party = [int(line.strip()) for line in lines]
        party = np.array(party)

    with open("modded_data/mpsex.dat") as f:
        lines = f.readlines()
        mpsex = [int(line.strip()) for line in lines]
        mpsex = np.array(mpsex)

    with open("data_lab2/mpdistrict.dat") as f:
        lines = f.readlines()
        mpdistr = [int(line.strip()) for line in lines]
        mpdistr = np.array(mpdistr)

    epochs = 300
    weights = np.random.normal(size=(10 ,10, 31))
    pos = votes_of_mps(votes, weights, epochs)
    pos = np.array(pos)

    party_names = {0:'No party', 1:'M', 2:'Fp', 3:'S', 4:'V', 5:'MP', 6:'KD', 7:'C'}
    sex_names = {0:'Male', 1:'Female'}
    distr_names = {}
    for i in range(1,30):
        distr_names[i] = "District: {}".format(i)
    party_votes = generate_dict(pos, party_names, party)
    sex_votes = generate_dict(pos, sex_names, mpsex)
    distr_votes = generate_dict(pos, distr_names, mpdistr)

    fig1 = plt.figure()
    for i, p in enumerate(party_votes.values()):
        fig1.add_subplot(2,4,i+1)
        plt.imshow(p/np.sum(p), cmap='jet', vmin=0, vmax=1)
        #norm=colors.LogNorm(vmin=0.01, vmax=party_votes[0].max())
        plt.title(party_names[i])

    fig2 = plt.figure()
    for i, p in enumerate(sex_votes.values()):
        fig2.add_subplot(1,2,i+1)
        plt.imshow(p/np.sum(p), cmap='jet', vmin=0, vmax=1)
        #norm=colors.LogNorm(vmin=0.01, vmax=party_votes[0].max())
        plt.title(sex_names[i])

    fig3 = plt.figure()
    for i, p in enumerate(distr_votes.values()):
        fig3.add_subplot(5,6,i+1)
        plt.imshow(p/np.sum(p), cmap='jet', vmin=0, vmax=1)
        #norm=colors.LogNorm(vmin=0.01, vmax=party_votes[0].max())
        plt.title(distr_names[i+1])
    plt.show()
