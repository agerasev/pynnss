#!/usr/bin/python3

from node import Node

# Network is a Node that contains other Nodes connected with each other with Paths

class Path:
	def __init__(self, src, dst):
		self.src = src
		self.dst = dst
		
	def __repr__(self):
		return '{src: ' + str(self.src) + ', dst: ' + str(self.dst) + '}'

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
		self.paths = []
		self._forwardPaths = {}
		self._backwardPaths = {}

	def buildPaths(self):
		for path in self.paths:
			self._forwardPaths[path.src] = path
			self._backwardPaths[path.dst] = path

	def __repr__(self):
		rstr = ''
		rstr += '{\n'
		rstr += 'Node: ' + Node.__repr__(self) + ',\n'
		rstr += 'nodes: ' + str(self.nodes) + ',\n'
		rstr += 'paths: ' + str(self.paths) + ',\n'
		rstr += '_forwardPaths: ' + str(self._forwardPaths) + ',\n'
		rstr += '_backwardPaths: ' + str(self._backwardPaths) + ',\n'
		rstr += '}'
		return rstr