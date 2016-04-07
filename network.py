#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.node import Node, NodeInfo


class Path:
	def __init__(self, src, dst, size, mem=False):
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

		def newMemory(self, factory):
			nodes = self._fornodes(lambda n: n.newMemory(factory))
			paths = self._forpaths(lambda p: factory.copy(p))
			return self._Memory(nodes, paths)

		class _Error(Node._State._Error, _Nodes, _Paths):
			def __init__(self, nodes, paths):
				_Nodes.__init__(self, nodes)
				_Paths.__init__(self, paths)
				Node._State._Error.__init__(self)

		def newError(self, factory):
			nodes = self._fornodes(lambda n: n.newError(factory))
			paths = self._forpaths(lambda p: factory.zeros(p.shape))
			return self._Error(nodes, paths)

		class _Gradient(Node._State._Gradient, _Nodes):
			def __init__(self, nodes):
				_Nodes.__init__(self, nodes)
				Node._State._Gradient.__init__(self)

		def newGradient(self, factory):
			nodes = self._fornodes(lambda n: n.newGradient(factory))
			return self._Gradient(nodes)

		class _Rate(Node._State._Rate, _Nodes):
			def __init__(self, nodes):
				_Nodes.__init__(self, nodes)
				Node._State._Rate.__init__(self)

		def newRate(self, factory, *args, **kwargs):
			nodes = self._fornodes(
				lambda n: n.newRate(factory, *args, **kwargs)
				)
			return self._Rate(nodes)

		def __init__(self, nodes, paths):
			_Nodes.__init__(self, nodes)
			_Paths.__init__(self, paths)
			Node._State.__init__(self)

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

	def newTrace(self, factory):
		return self._Trace([n.newTrace(factory) for n in self.nodes])

	class _Context(Node._Context, _Nodes, _Paths):
		class _Src:
			def __init__(self, outer):
				pass

		class _Dst:
			def __init__(self, outer):
				pass

		def __init__(self, node, nodes, paths):
			_Nodes.__init__(self, nodes)
			_Paths.__init__(self, paths)
			Node._Context.__init__(self, node)

	def newContext(self, factory):
		nodes = [n.newContext(factory) for n in self.nodes]
		paths = [factory.empty(p.size) for p in self.paths]
		return self._Context(self, nodes, paths)

	def __init__(self, isizes, osizes, **kwargs):
		_Nodes.__init__(self)
		_Paths.__init__(self)
		Node.__init__(self, isizes, osizes, **kwargs)
		NetworkInfo.__init__(self)
		self.ipaths = [None]*self.inum
		self.opaths = [None]*self.onum
		self._flink = {}
		self._blink = {}

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

	def connect(self, connids, mem=False):
		if connids is tuple:
			connids = [connids]
		for sid, did in connids:
			src, snode, ssize = self._snodeid(sid)
			dst, dnode, dsize = self._dnodeid(did)
			if ssize != dsize:
				raise Exception('sizes dont match')
			self._sisfree(src)
			self._disfree(dst)
			self.paths.append(Path(src, dst, ssize, mem))
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
