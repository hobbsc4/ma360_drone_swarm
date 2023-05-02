from swarm_model import SwarmModel
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import itertools

@jit(forceobj=True)
def calc(drone_range, num_trials, weights):
    model = SwarmModel(weights, 0)
    returns = []
    pbar = tqdm(total =len(drone_range) * num_trials)
    for number in drone_range:
        trials = np.zeros(num_trials)
        for i in range(num_trials):
            trials[i] = model.run(weights, number)
            pbar.update()
        returns.append(trials.mean())
    return returns

def saveplot(drone_range, returns, params):
    plt.plot(drone_range, returns)
    plt.title("Utility vs. Number of Drones")
    plt.xlabel("Number of Drones")
    plt.ylabel("Utitlity (USD)")
    #plt.savefig("returns/plots/" + params + " plot.png")
    #plt.clf()

def savevalues(drone_range, returns, params):
    np.savetxt("returns/values/" + params + " values.txt", (drone_range, returns), fmt="%d")

def main():
    n_drones = list(range(1, 101, 3))
    num_trials = 3
    param_range = itertools.product([0.5, 1, 2], repeat=5)

    for i in (list(param_range)[18:]):
        objvs = calc(n_drones, num_trials, i)
        #saveplot(n_drones, objvs, str(i))
        savevalues(n_drones, objvs, str(i))



    #objvs = calc(n_drones, num_trials, [1,1,1,1,1])
    #saveplot(n_drones, objvs, "test")
    #savevalues(n_drones, objvs, "test")



if __name__ == "__main__":
    main()