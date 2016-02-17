#!/usr/bin/python3

from node import ConvNode
from network import Network, Path

net = Network(1,1)
net.nodes[1] = ConvNode(4,4);
net.paths.append(Path((0,0), (1,0)))
net.paths.append(Path((1,0), (0,0)));
print(net)
