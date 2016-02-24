#!/usr/bin/python3

from node import Node
from path import Pipe

# Network is a Node that contains other Nodes connected with each other with Paths

class Network(Node):
	def __init__(self, nins, nouts):
		Node.__init__(self, nins, nouts)
		self.ins = [None]*nins
		self.outs = [None]*nouts
		self.nodes = {}
		self.paths = []
		self._fpath_cache = {}
		self._bpath_cache = {}

	def cache(self):
		for i in range(len(self.paths)):
			p = self.paths[i]
			self._fpath_cache[p.src] = i
			self._bpath_cache[p.dst] = i

	class _Memory(Node._Memory):
		def __init__(self):
			Node._Memory.__init__(self)
			self.nodes = {}
			self.pipes = []


	def Memory(self):
		mem = Network._Memory()
		for key in self.nodes:
			mem.nodes[key] = self.nodes[key].Memory()
		for i in range(len(self.paths)):
			mem.pipes.append(Pipe())
		return mem

	def _MemNext(self, mem):
		nmem = Network._Memory()
		for key in self.nodes:
			nmem.nodes[key] = None
		for i in range(len(mem.pipes)):
			nmem.pipes.append(Pipe(mem.pipes[i].data))
		return nmem

	def step(self, mem, nmem):
		found = False
		for key in self.nodes:
			vins = []
			node = self.nodes[key]
			for i in range(node.nins):
				pipe = nmem.pipes[self._bpath_cache[(key, i)]]
				if pipe.data is not None:
					vins.append(pipe.data)
			if len(vins) != node.nins:
				continue
			if nmem.nodes[key] is not None:
				raise Exception('Node activated twice')
			(nmem.nodes[key], vouts) = node.feedforward(mem.nodes[key], vins)
			for i in range(node.nins):
				nmem.pipes[self._bpath_cache[(key, i)]].data = None
			for i in range(node.nouts):
				pipe = nmem.pipes[self._fpath_cache[(key, i)]]
				if pipe.data is not None:
					raise Exception('Output pipe is not empty')
				pipe.data = vouts[i]
			found = True
		return found

	def _feedforward(self, mem, vins):
		nmem = self._MemNext(mem)
		for i in range(self.nins):
			nmem.pipes[self._fpath_cache[(-1, i)]].data = vins[i]
		found = True
		while found:
			found = self.step(mem, nmem)
		vouts = []
		for i in range(self.nouts):
			pipe = nmem.pipes[self._bpath_cache[(-1, i)]]
			if pipe.data is None:
				raise Exception('Output is None, possibly caused by wrong structure')
			vouts.append(pipe.data)
			pipe.data = None
		return (nmem, vouts)


