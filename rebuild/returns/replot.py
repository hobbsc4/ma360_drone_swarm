from matplotlib import pyplot as plt


param = (0.5, 0.5, 0.5, 0.5, 1)

with open("values/" + str(param) + " values.txt") as f:
    plt.plot(list(map(int, f.readline().split())), list(map(int, f.readline().split())))
    plt.show()