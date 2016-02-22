#!/usr/bin/python3

from element import Product, Sigmoid

import math
from random import random
import numpy as np

node = Product(4, 2)
sigma = Sigmoid(2)

# print('weight:\n' + str(node.weight))
# print('bias: ' + str(node.bias))

batch_size = 0x10
batches_num = 0x100

for k in range(0x10):
	cost = 0.0

	for j in range(batches_num):
		nmems = [node.Memory()]
		smems = [sigma.Memory()]
		nexp = node.Experience()
		sexp = sigma.Experience()

		for i in range(batch_size):
			a = math.floor(random()*4)
			lin = [0,0,0,0]
			lin[a] = 1
			vins = [np.array(lin)]
			
			# feedforward
			(mem, vouts) = node.step(nmems[len(nmems) - 1], vins)
			nmems.append(mem)
			(mem, _) = sigma.step(smems[len(smems) - 1], vouts)
			smems.append(mem)

			lres = [0,0]
			lres[a%2] = 1
			vres = np.array(lres)
			verrs = [vouts[0] - vres]
			cost += np.sum((verrs[0])**2)

			# backpropagate
			verrs = sigma.backprop(sexp, smems.pop(), verrs)
			node.backprop(nexp, nmems.pop(), verrs)

		node.learn(nexp, 1e-2/batch_size)

	print(str(k) + ' cost: ' + str(cost/batch_size/batches_num))
	