#!/usr/bin/python3

from node import Node

from path import Pipe

# Network is a Node that contains other Nodes connected with each other with Paths

class Network(Node):
	nodes = {}
	paths = []
	_fpath_cache = {}
	_bpath_cache = {}

	def __init__(self, nin, nout):
		Node.__init__(self, nin, nout)
		self.ins = [None]*nin
		self.outs = [None]*nout

	def cache(self):
		for i in range(len(self.paths)):
			p = self.paths[i]
			self._fpath_cache[p.src] = i
			self._bpath_cache[p.dst] = i

	class _Memory(Node._Memory):
		nodes = {}
		pipes = []
		def __init__(self, nins, nouts):
			Node._Memory.__init__(self, nins, nouts)


	def Memory(self):
		mem = Network._Memory(self.nins, self.nouts)
		for k, v in self.nodes:
			mem.nodes[k] = v.Memory()
		for p in self.paths:
			mem.pipes.append(Pipe())

	def _nextmem(self, mem):
		nmem = Network._Memory(self.nins, self.nouts)
		for p in mem.pipes:
			nmem.paths.append(Pipe(p))

	def _step(self, mem, vins):
		nmem = self._nextmem(mem)
		return (nmem, None)
