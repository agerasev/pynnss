#!/usr/bin/python3

from path import Path
from node import ProductNode, SigmoidNode
from network import Network

net = Network(1,1)

net.nodes.append(ProductNode(3,5))
net.nodes.append(SigmoidNode(5))
net.paths.append(Path(net.nodes[0].outs[0], net.nodes[1].ins[0]))
net.ins[0] = net.nodes[0].ins[0]
net.outs[0] = net.nodes[1].outs[0]

import numpy as np

net.ins[0].data = np.array([1,0,0])

net.step()

print('Network: ' + str(net))

net.step()

print('Network: ' + str(net))
