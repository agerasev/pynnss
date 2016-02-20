#!/usr/bin/python3

from element import Product

import numpy as np

node = Product(6, 5)

(mem, vouts) = node.step(node.Memory(), [np.array([1, 0, 0, 1, 0, 0])])

print('vouts: ' + str(vouts))