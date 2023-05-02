from matplotlib import pyplot as plt
import itertools
import numpy as np

params = itertools.product([0.5, 1, 2], repeat=5)
integrals = []

for param in params:

    with open("values/" + str(param) + " values.txt") as f:

        ndrones = list(map(int, f.readline().split()))
        objv = list(map(int, f.readline().split()))

        integral = int(np.mean(objv))
        integrals.append((integral, param))
        #plt.figtext(0.15, 0.83, integral)

        #plt.title("Utility vs. Number of Drones")
        #plt.xlabel("Number of Drones")
        #plt.ylabel("Utitlity (USD)")
        #plt.xlim([0, 100])
        #plt.ylim([-0.4e6, 3e6])

        #plt.plot(ndrones, objv)
        #plt.savefig("plots/" + str(param) + " vomitplot.png")
        #plt.clf()
        f.close()

integrals.sort(reverse = True)
with open("integrals.txt", 'w') as l:
    for i in integrals:
        l.write(str(i) + "\n")
    l.close()
