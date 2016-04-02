#!/usr/bin/python3

import numpy as np


class Array:
	def __init__(self, size):
		self.size = size
		self.data = np.zeros(size)


# apply func for arrays or sequences of arrays
def _unwrap(func, *args, opt=()):
	if isinstance(args[0], Array):
		return func(*args, *opt)
	else:
		lst = []
		for a in zip(*args):
			lst.append(func(*a, *opt))
		return tuple(lst)


# copy one array to another
def _copyto(dst, src):
	np.copyto(dst.data, src.data)


def copyto(dst, src):
	_unwrap(_copyto, dst, src)


# create new array and copy to it
def _copy(arr):
	ret = Array(arr.size)
	_copyto(ret, arr)
	return ret


def copy(arr):
	return _unwrap(_copy, arr)


# add one array to another
def _addto(dst, src):
	dst.data += src.data


def addto(dst, src):
	_unwrap(_addto, dst, src)

# unit test
if __name__ == '__main__':
	print('unit test:')

	print('copyto ... ', end='')
	a = Array(4),
	a[0].data[0] = 1
	b = Array(4),
	copyto(b, a)
	a[0].data[0] = 2
	assert(b[0].data[0] == 1)
	print('ok')

	print('copy ... ', end='')
	a = Array(4), Array(2)
	a[0].data[3] = 3
	a[1].data[1] = 1
	b = copy(a)
	assert(b[0].data[3] == 3 and b[1].data[1] == 1)
	print('ok')

	print('addto ... ', end='')
	a = Array(4),
	a[0].data[0] = 1
	b = Array(4),
	b[0].data[0] = 2
	addto(b, a)
	assert(b[0].data[0] == 3)
	print('ok')
