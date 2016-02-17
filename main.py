#!/usr/bin/python3

# Signal is portion of data NNs operate with

class Signal:
	data = None

# Node is structural unit of NN

class NodeError(Exception):
	def __init__(self, value):
		Exception.__init__(self)
		self.value = value
	def __str__(self):
		return repr(self.value)

class Node:
	def __init__(self, nin, nout):
		self.inputs = [None]*nin
		self.outputs = [None]*nout

	def _step(self):
		raise NotImplementedError()

	def step(self):
		# check at least one input is not None
		found = False
		for signal in self.inputs:
			if signal is not None:
				found = True
				break
		if not found:
			return False

		# check all outputs is None
		found = False
		for signal in self.outputs:
			if signal is not None:
				found = True
				break
		if found:
			raise NodeError('outputs are not None : step will cause loss of data')

		# transmit signal
		self._step()
		return True



# Network is a Node that contains other Nodes connected with each other with Paths

class Path:
	def __init__(self, src, dst):
		self.src = src
		self.dst = dst

class NetworkError(Exception):
	def __init__(self, value):
		Exception.__init__(self)
		self.value = value
	def __str__(self):
		return repr(self.value)

class Network(Node):
	def __init__(self, nin, nout):
		Node.__init__(self, nin, nout)
		self.nodes = {}
		self._paths = []
		self._forwardPaths = {}
		self._backwardPaths = {}

	def buildPaths(self):
		# TODO: build paths
		pass
