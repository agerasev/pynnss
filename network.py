#!/usr/bin/python3

from node import Node

# Network is a Node that contains other Nodes connected with each other with Paths

class Network(Node):
	def __init__(self, nin, nout):
		Node.__init__(self, nin, nout)
		self.nodes = {}
		self.paths = {}

	

	def _step(self):
		for path in self.paths:
			path.shift()
		for node in self.nodes:
			node.step()

	def clear(self):
		for path in self.paths:
			path.clear()
		for node in self.nodes:
			node.clear()