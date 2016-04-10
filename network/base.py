#!/usr/bin/python3

from pynn import array


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

	def _copynodes(self, out):
		for ns, no in zip(self.nodes, out.nodes):
			if ns is not None:
				ns.copyto(no)


class _Paths:
	def __init__(self, paths=None):
		self.paths = _list(paths)

	def _forpaths(self, fmap):
		return _foreach(self.paths, fmap)

	def _copypaths(self, out):
		for ps, po in zip(self.paths, out.paths):
			if ps is not None:
				array.copy(po, ps)
