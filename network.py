#!/usr/bin/python3

from pynn.node import Node
from pynn.path import Pipe

# Network is a Node that contains other Nodes connected with each other with Paths

class Network(Node):
	def __init__(self, nins, nouts):
		Node.__init__(self, nins, nouts)
		self.ins = [None]*nins
		self.outs = [None]*nouts
		self.nodes = {}
		self.paths = []
		self._fpath_link = {}
		self._bpath_link = {}

	def update(self):
		for i in range(len(self.paths)):
			p = self.paths[i]
			self._fpath_link[p.src] = i
			self._bpath_link[p.dst] = i

	class _Memory(Node._Memory):
		def __init__(self):
			Node._Memory.__init__(self)
			self.nodes = {}
			self.pipes = []
		
		def next(self):
			nmem = Network._Memory()
			for key in self.nodes:
				nmem.nodes[key] = None
			for i in range(len(self.pipes)):
				nmem.pipes.append(Pipe(self.pipes[i].data))
			return nmem

	def Memory(self):
		mem = Network._Memory()
		for key in self.nodes:
			mem.nodes[key] = self.nodes[key].Memory()
		for i in range(len(self.paths)):
			mem.pipes.append(Pipe())
		return mem

	def step(self, mem, nmem):
		found = False
		for key in self.nodes:
			vins = []
			node = self.nodes[key]
			for i in range(node.nins):
				pipe = nmem.pipes[self._bpath_link[(key, i)]]
				if pipe.data is not None:
					vins.append(pipe.data)
			if len(vins) != node.nins:
				continue
			if nmem.nodes[key] is not None:
				raise Exception('Node activated twice')
			(nmem.nodes[key], vouts) = node.feedforward(mem.nodes[key], vins)
			for i in range(node.nins):
				nmem.pipes[self._bpath_link[(key, i)]].data = None
			for i in range(node.nouts):
				pipe = nmem.pipes[self._fpath_link[(key, i)]]
				if pipe.data is not None:
					raise Exception('Output pipe is not empty')
				pipe.data = vouts[i]
			found = True
		return found

	def _feedforward(self, mem, vins):
		nmem = mem.next()
		for i in range(self.nins):
			nmem.pipes[self._fpath_link[(-1, i)]].data = vins[i]
		found = True
		while found:
			found = self.step(mem, nmem)
		vouts = []
		for i in range(self.nouts):
			pipe = nmem.pipes[self._bpath_link[(-1, i)]]
			if pipe.data is None:
				raise Exception('Output is empty, possibly caused by wrong structure')
			vouts.append(pipe.data)
			pipe.data = None
		return (nmem, vouts)

	class _Experience(Node._Experience):
		def __init__(self):
			Node._Experience.__init__(self)
			self.nodes = {}
			self.pipes = []

		def clip(self, val):
			for key in self.nodes:
				self.nodes[key].clip(val)

	def Experience(self):
		exp = Network._Experience()
		for key in self.nodes:
			exp.nodes[key] = self.nodes[key].Experience()
		for i in range(len(self.paths)):
			exp.pipes.append(Pipe())
		return exp

	def backstep(self, exp, mem):
		found = False
		for key in self.nodes:
			eouts = []
			node = self.nodes[key]
			for i in range(node.nouts):
				pipe = exp.pipes[self._fpath_link[(key, i)]]
				if pipe.data is not None:
					eouts.append(pipe.data)
			if len(eouts) != node.nouts:
				continue
			if exp.nodes[key].count > exp.count:
				raise Exception('Node activated twice')
			eins = node.backprop(exp.nodes[key], mem.nodes[key], eouts)
			for i in range(node.nouts):
				exp.pipes[self._fpath_link[(key, i)]].data = None
			for i in range(node.nins):
				pipe = exp.pipes[self._bpath_link[(key, i)]]
				if pipe.data is not None:
					raise Exception('Input pipe is not empty')
				pipe.data = eins[i]
			found = True
		return found

	def _backprop(self, exp, mem, eouts):
		for i in range(self.nouts):
			pipe = exp.pipes[self._bpath_link[(-1, i)]]
			if pipe.data is not None:
				raise Exception('Output is not empty')
			pipe.data = eouts[i]
		found = True
		while found:
			found = self.backstep(exp, mem)
		eins = []
		for i in range(self.nins):
			pipe = exp.pipes[self._fpath_link[(-1, i)]]
			if pipe.data is None:
				raise Exception('Input is empty, possibly caused by wrong structure')
			eins.append(pipe.data)
			pipe.data = None
		return eins

	def learn(self, exp, rate):
		for key in self.nodes:
			self.nodes[key].learn(exp.nodes[key], rate)
