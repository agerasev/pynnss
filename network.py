#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.node import Node, NodeInfo


class Path:
	def __init__(self, src, dst, size=-1, mem=False):
		self.src = src
		self.dst = dst
		self.size = size
		self.mem = mem


def _list(lst):
	if lst is not None:
		return list(lst)
	else:
		return []


def _foreach(lst, fmap):
	res = []
	for item in lst:
		out = None
		if item is not None:
			out = fmap(item)
		res.append(out)
	return res


class _Nodes:
	def __init__(self, nodes=None):
		self.nodes = _list(nodes)

	def _fornodes(self, fmap):
		return _foreach(self.nodes, fmap)


class _Paths:
	def __init__(self, paths=None):
		self.paths = _list(paths)

	def _forpaths(self, fmap):
		return _foreach(self.paths, fmap)


class NetworkInfo(NodeInfo, _Nodes, _Paths):
	def __init__(self):
		pass


class Network(Node, NetworkInfo, _Nodes, _Paths):
	class _State(Node._State, _Nodes, _Paths):
		class _Memory(Node._State._Memory, _Nodes, _Paths):
			def __init__(self, nodes, paths):
				_Nodes.__init__(self, nodes)
				_Paths.__init__(self, paths)
				Node._State._Memory.__init__(self)

			def copyto(self, out):
				for ns, no in zip(self.nodes, out.nodes):
					if ns is not None:
						ns.copyto(no)
				for ps, po in zip(self.paths, out.paths):
					if ps is not None:
						array.copy(po, ps)

		def newMemory(self, factory):
			nodes = self._fornodes(lambda n: n.newMemory(factory))
			paths = self._forpaths(lambda p: factory.copy(p))
			return self._Memory(nodes, paths)

		class _Error(Node._State._Error, _Nodes, _Paths):
			def __init__(self, nodes, paths):
				_Nodes.__init__(self, nodes)
				_Paths.__init__(self, paths)
				Node._State._Error.__init__(self)

			def copyto(self, out):
				for ns, no in zip(self.nodes, out.nodes):
					if ns is not None:
						ns.copyto(no)
				for ps, po in zip(self.paths, out.paths):
					if ps is not None:
						array.copy(po, ps)

		def newError(self, factory):
			nodes = self._fornodes(lambda n: n.newError(factory))
			paths = self._forpaths(lambda p: factory.zeros(p.shape))
			return self._Error(nodes, paths)

		class _Gradient(Node._State._Gradient, _Nodes):
			def __init__(self, nodes):
				_Nodes.__init__(self, nodes)
				Node._State._Gradient.__init__(self)

			def mul(self, factor):
				self._fornodes(lambda n: n.mul(factor))

			def clip(self, value):
				self._fornodes(lambda n: n.clip(value))

			def clear(self):
				self._fornodes(lambda n: n.clear())

		def newGradient(self, factory):
			nodes = self._fornodes(lambda n: n.newGradient(factory))
			return self._Gradient(nodes)

		class _Rate(Node._State._Rate, _Nodes):
			def __init__(self, nodes):
				_Nodes.__init__(self, nodes)
				Node._State._Rate.__init__(self)

		class _RateAdaGrad(_Rate):
			def __init__(self, nodes):
				Network._State._Rate.__init__(self, nodes)

			def update(self, grad):
				for nr, ng in zip(self.nodes, grad.nodes):
					if nr is not None and ng is not None:
						nr.update(ng)

		def newRate(self, factory, *args, **kwargs):
			nodes = self._fornodes(
				lambda n: n.newRate(factory, *args, **kwargs)
				)
			if kwargs.get('adagrad', False):
				return self._RateAdaGrad(nodes)
			else:
				return self._Rate(nodes)

		def __init__(self, nodes, paths):
			_Nodes.__init__(self, nodes)
			_Paths.__init__(self, paths)
			Node._State.__init__(self)

		def learn(self, grad, rate):
			for n, g, r in zip(self.nodes, grad.nodes, rate.nodes):
				if n is not None and g is not None:
					n.learn(g, r)

	def newState(self, factory):
		nodes = [n.newState(factory) for n in self.nodes]
		paths = []
		for path in self.paths:
			data = None
			if path.mem:
				data = factory.zeros(path.size)
			paths.append(data)
		return self._State(nodes, paths)

	class _Trace(Node._Trace, _Nodes):
		def __init__(self, nodes):
			_Nodes.__init__(self, nodes)
			Node._Trace.__init__(self)

		def copyto(self, out):
			for ns, no in zip(self.nodes, out.nodes):
				if ns is not None:
					ns.copyto(no)

	def newTrace(self, factory):
		return self._Trace([n.newTrace(factory) for n in self.nodes])

	class _Context(Node._Context, _Nodes, _Paths):
		class _Srcs:
			def __init__(self, outer):
				self.outer = outer

			def __getitem__(self, key):
				return self.outer._srcs[key]

			def __setitem__(self, key, value):
				self.outer._srcs[key] = value
				did = self.outer.node.ipaths[key].dst
				self.outer.nodes[did[0]].srcs[did[1]] = value

		class _Dsts:
			def __init__(self, outer):
				self.outer = outer

			def __getitem__(self, key):
				return self.outer._dsts[key]

			def __setitem__(self, key, value):
				self.outer._dsts[key] = value
				sid = self.outer.node.opaths[key].src
				self.outer.nodes[sid[0]].dsts[sid[1]] = value

		def _ssrcs(self, srcs):
			for i in range(self.node.inum):
				self.srcs[i] = srcs[i]

		def _gsrcs(self):
			return self._Srcs(self)

		srcs = property(_gsrcs, _ssrcs)

		def _sdsts(self, dsts):
			for i in range(self.node.onum):
				self.dsts[i] = dsts[i]

		def _gdsts(self):
			return self._Dsts(self)

		dsts = property(_gdsts, _sdsts)

		def _gstate(self):
			return self._state

		def _sstate(self, state):
			if state is None:
				nss = [None]*len(self.nodes)
			else:
				nss = state.nodes
			for nc, ns in zip(self.nodes, nss):
				if nc is not None:
					nc.state = ns
			self._state = state

		state = property(_gstate, _sstate)

		def _gtrace(self):
			return self._trace

		def _strace(self, trace):
			if trace is None:
				nts = [None]*len(self.nodes)
			else:
				nts = trace.nodes
			for nc, nt in zip(self.nodes, nts):
				if nc is not None:
					nc.trace = nt
			self._trace = trace

		trace = property(_gtrace, _strace)

		def _ggrad(self):
			return self._grad

		def _sgrad(self, grad):
			if grad is None:
				ngs = [None]*len(self.nodes)
			else:
				ngs = grad.nodes
			for nc, ng in zip(self.nodes, ngs):
				if nc is not None:
					nc.grad = ng
			self._grad = grad

		grad = property(_ggrad, _sgrad)

		def _grate(self):
			return self._rate

		def _srate(self, rate):
			if rate is None:
				nrs = [None]*len(self.nodes)
			else:
				nrs = rate.nodes
			for nc, nr in zip(self.nodes, nrs):
				if nc is not None:
					nc.rate = nr
			self._rate = rate

		rate = property(_grate, _srate)

		def __init__(self, node, nodes, paths):
			_Nodes.__init__(self, nodes)
			_Paths.__init__(self, paths)
			self._srcs = [None]*node.inum
			self._dsts = [None]*node.onum
			Node._Context.__init__(self, node)
			for path, arr in zip(self.node.paths, self.paths):
				src = path.src
				dst = path.dst
				self.nodes[src[0]].dsts[src[1]] = arr
				self.nodes[dst[0]].srcs[dst[1]] = arr

	def newContext(self, factory):
		nodes = [n.newContext(factory) for n in self.nodes]
		paths = [factory.empty(p.size) for p in self.paths]
		return self._Context(self, nodes, paths)

	def _gisize(self):
		return self.isizes[0]

	isize = property(_gisize)

	def _gosize(self):
		return self.osizes[0]

	osize = property(_gosize)

	def __init__(self, isizes, osizes, **kwargs):
		_Nodes.__init__(self)
		_Paths.__init__(self)
		Node.__init__(self, isizes, osizes, **kwargs)
		NetworkInfo.__init__(self)
		self.ipaths = [None]*self.inum
		self.opaths = [None]*self.onum
		self._flink = {}
		self._blink = {}
		self.order = None

	def addnodes(self, nodes):
		if isinstance(nodes, Node):
			nodes = [nodes]
		for node in nodes:
			self.nodes.append(node)

	def _nodeid(self, nid):
		pos = 0
		if type(nid) == tuple:
			key = nid[0]
			pos = nid[1]
		else:
			key = nid
		if key < 0 or key >= len(self.nodes):
			raise Exception('no node with key %d' % key)
		node = self.nodes[key]
		return (key, pos), node

	def _snodeid(self, sid):
		(key, pos), node = self._nodeid(sid)
		if pos < 0 or pos >= node.onum:
			raise Exception('wrong opos %d for node %d' % (pos, key))
		return (key, pos), node, node.osizes[pos]

	def _dnodeid(self, did):
		(key, pos), node = self._nodeid(did)
		if pos < 0 or pos >= node.inum:
			raise Exception('wrong ipos %d for node %d' % (pos, key))
		return (key, pos), node, node.isizes[pos]

	def _sisfree(self, src):
		if src in self._flink.keys():
			raise Exception('output (%d,%d) already connected' % src)

	def _disfree(self, dst):
		if dst in self._blink.keys():
			raise Exception('input (%d,%d) already connected' % dst)

	def connect(self, paths):
		if isinstance(paths, Path):
			paths = [paths]
		for path in paths:
			src, snode, ssize = self._snodeid(path.src)
			dst, dnode, dsize = self._dnodeid(path.dst)
			if ssize != dsize:
				raise Exception('sizes dont match')
			self._sisfree(src)
			self._disfree(dst)
			self.paths.append(Path(src, dst, ssize, path.mem))
			self._flink[src] = len(self.paths) - 1
			self._blink[dst] = len(self.paths) - 1

	def setinputs(self, dids):
		if type(dids) is int or type(dids) is tuple:
			dids = [dids]
		for i, did in enumerate(dids):
			dst, dnode, dsize = self._dnodeid(did)
			self._disfree(dst)
			if dsize != self.isizes[i]:
				raise Exception('sizes dont match')
			self.ipaths[i] = Path(None, dst, dsize)
			self._blink[dst] = -1

	def setoutputs(self, sids):
		if type(sids) is int or type(sids) is tuple:
			sids = [sids]
		for i, sid in enumerate(sids):
			src, snode, ssize = self._snodeid(sid)
			self._sisfree(src)
			if ssize != self.osizes[i]:
				raise Exception('sizes dont match')
			self.opaths[i] = Path(src, None, ssize)
			self._flink[src] = -1

	def _transmit(self, ctx):
		if self.order is None:
			self.order = list(range(len(self.nodes)))
		# TODO: copy nodes too
		for pc, pm in zip(ctx.paths, ctx.mem.paths):
			if pm is not None:
				array.copy(pc, pm)
		znc = list(zip(self.nodes, ctx.nodes))
		for i in self.order:
			n, nc = znc[i]
			n.transmit(nc)
		for pc, pm in zip(ctx.paths, ctx.mem.paths):
			if pm is not None:
				array.copy(pm, pc)

	def _backprop(self, ctx):
		if self.order is None:
			self.order = list(range(len(self.nodes)))
		for pc, pe in zip(ctx.paths, ctx.err.paths):
			if pe is not None:
				array.copy(pc, pe)
		znc = list(zip(self.nodes, ctx.nodes))
		for i in reversed(self.order):
			n, nc = znc[i]
			n.backprop(nc)
		for pc, pe in zip(ctx.paths, ctx.err.paths):
			if pe is not None:
				array.copy(pe, pc)
