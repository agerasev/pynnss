#!/usr/bin/python3

from element import Product, Sigmoid
from path import Path
from network import Network

# import math
# from random import random
import numpy as np

net = Network(1, 1)

net.nodes[0] = Product(4, 2)
net.nodes[1] = Sigmoid(2)

net.paths.append(Path((-1, 0), (0, 0)))
net.paths.append(Path((0, 0), (1, 0)))
net.paths.append(Path((1, 0), (-1, 0)))

net.update()

vins = [np.array([0,0,0,0])]
(mem, vouts) = net.feedforward(net.Memory(), vins)
print(str(vouts))
