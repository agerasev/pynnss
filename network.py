#!/usr/bin/python3

import numpy as np
import pynn.array as array
from pynn.node import Node, Site


class Path:
	def __init__(self, src, dst, site, mem=False):
		self.src = src
		self.dst = dst
		self.info = site
		self.mem = mem


class _Nodes:
	def __init__(self, nodes=None):
		self.nodes = {}
		if nodes is not None:
			for key, node in nodes.items():
				if node is not None:
					self.nodes[key] = node


class _Paths:
	def __init__(self, paths=None):
		self.paths = {}
		if paths is not None:
			for key, path in paths.itemss():
				if path is not None:
					self.paths[key] = path

	def _addpath(self, path):
		key = len(self.paths)
		self.paths[key] = path
		return key


class Network(Node, _Nodes, _Paths):
	class _State(Node._State, _Nodes, _Paths):
		def __init__(self, nodes, paths):
			_Nodes.__init__(nodes)
			_Paths.__init__(paths)
			Node._State.__init__(self)

		class _Memory(Node._State._Memory, _Nodes, _Paths):
			def __init__(self, nodes, paths):
				_Nodes.__init__(self, nodes)
				_Paths.__init__(self, nodes)
				Node._State._Memory.__init__(self)

		def newMemory(self, factory):
			nodes = {k: n.newMemory(factory) for k, n in self.nodes.items()}
			paths = {k: factory.copy(p) for k, p in self.paths.items()}
			return self._Memory(nodes, paths)

		class _Error(Node._State._Error, _Nodes, _Paths):
			def __init__(self, nodes, paths):
				_Nodes.__init__(self, nodes)
				_Paths.__init__(self, nodes)
				Node._State._Error.__init__(self)

		def newError(self, factory):
			nodes = {k: n.newError(factory) for k, n in self.nodes.items()}
			paths = {k: factory.zeros(p.shape) for k, p in self.paths.items()}
			return self._Error(nodes, paths)

		class _Gradient(Node._State._Gradient, _Nodes):
			def __init__(self):
				pass

		class _Rate(Node._State._Rate, _Nodes):
			def __init__(self):
				pass

	def newState(self, factory):
		nodes = {k: n.newState() for k, n in self.nodes.items()}
		paths = {}
		for key, path in self.paths:
			info = path.info
			if path.mem:
				paths[key] = factory.zeros(info.size)
		return self._State(nodes, paths)

	class _Trace(Node._Trace, _Nodes):
		def __init__(self, nodes):
			_Nodes.__init__(self, nodes)
			Node._Trace.__init__(self)

	def newTrace(self, factory):
		nodes = {k: n.newTrace() for k, n in self.nodes.items()}
		return self._Trace(nodes)

	class _Context(Node._Context, _Nodes, _Paths):
		def _gmem(self):
			return self._mem

		def _smem(self, mem):
			self._mem = mem
			if mem is not None:
				for k, n in self.nodes.items():
					n.mem = mem.nodes[k]
				for k, p in self.paths.items():
					pass

		def __init__(self, nodes, paths, ipaths, opaths, src, dst, **kwargs):
			_Nodes.__init__(nodes)
			_Paths.__init__(paths)
			Node._Context.__init__(self, src, dst, **kwargs)

	def newContext(self, factory, *args, **kwargs):
		paths = {}
		for key, path in self.paths.items():
			if path.mem:
				paths[key] = path
			else:
				paths[key] = factory.empty(path.site.size)
		nodes = {}
		for key, node in self.nodes.items():
			if node.inum == 1:
				pass
			else:
				pass

		return self._Context(self, nodes, paths, *args, **kwargs)

	def __init__(self, isizes, osizes, **kwargs):
		_Nodes.__init__(self)
		_Paths.__init__(self)
		if type(isizes) == int:
			isites = [Site(isizes)]
		else:
			isites = []
			for size in isizes:
				isites.append(Site(size))
		if type(osizes) == int:
			osites = [Site(osizes)]
		else:
			osites = []
			for size in osizes:
				osites.append(Site(size))
		Node.__init__(self, isites, osites, **kwargs)
		self.ipaths = []
		self.opaths = []
		self._flink = {}
		self._blink = {}

	def add(self, key, node):
		if key in self.nodes.keys():
			raise Exception('key %d already used' % key)
		self.nodes[key] = node

	def _getnode(self, nid):
		sn = 0
		if type(nid) == tuple:
			key = nid[0]
			sn = nid[1]
		else:
			key = nid
		if key not in self.nodes.keys():
			raise Exception('no node with key %d' % key)
		node = self.nodes[key]
		return (key, sn), node

	def _getsnode(self, sid):
		(key, sn), node = self._getnode(sid)
		if sn < 0 or sn >= node.onum:
			raise Exception('wrong output site %d for node %d' % (sn, key))
		return (key, sn), node, node.osites[sn]

	def _getdnode(self, did):
		(key, sn), node = self._getnode(did)
		if sn < 0 or sn >= node.inum:
			raise Exception('wrong input site %d for node %d' % (sn, key))
		return (key, sn), node, node.isites[sn]

	def connect(self, sid, did, mem=False):
		src, snode, ssite = self._getsnode(sid)
		dst, dnode, dsite = self._getdnode(did)
		if ssite != dsite:
			raise Exception('sites not match')
		if src in self._flink.keys():
			raise Exception('output (%d,%d) already connected' % src)
		if dst in self._blink.keys():
			raise Exception('input (%d,%d) already connected' % dst)
		key = self.paths._addpath(Path(src, dst, ssite, mem))
		self._flink[src] = len(self.paths) - 1
		self._blink[dst] = len(self.paths) - 1
		return key

	def input(self, did):
		dst, dnode, dsite = self._getdnode(did)
		self.ipaths.append(Path(None, dst, dsite))

	def output(self, sid):
		src, snode, ssite = self._getsnode(sid)
		self.opaths.append(Path(src, None, ssite))
