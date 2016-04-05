#!/usr/bin/python3

import numpy as np
from pynn.network import Network
from pynn.element import Element


class Rate:
	def __init__(self, node, value):
		self.isnet = False

		if isinstance(node, Network):
			self.isnet = True
			net = node
			self.nodes = {}
			for key, node in net.nodes.items():
				rate = None
				if type(node) != Element or node.state is not None:
					rate = type(self)(node, value)
				self.nodes[key] = rate

		if isinstance(node, Element):
			self.value = value


class RateAdaGrad(Rate):
	def __init__(self, node, value):
		Rate.__init__(self, node, value)
		if not self.isnet:
			self.accum = np.zeros_like(node.state) + 1e-8
			self.factor = value
			self.value = self.factor/np.sqrt(self.accum)

	def update(self, grad):
		if grad is not None:
			if self.isnet:
				for key, rate in self.nodes.items():
					rate.update(grad.nodes[key])
			else:
				self.accum += grad.state**2
				self.value = self.factor/np.sqrt(self.accum)
