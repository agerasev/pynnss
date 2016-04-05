#!/usr/bin/python3

import numpy as np


class Array:
	def __init__(self, arg, dtype=None, gpu=False):
		self.gpu = gpu

		if type(arg) == Array:
			if dtype is None:
				dtype = arg.dtype
			self.data = np.array(arg.data, dtype=dtype)
		elif type(arg) == np.ndarray:
			if dtype is None:
				dtype = arg.dtype
			self.data = np.array(arg, dtype=dtype)
		elif type(arg) == tuple or type(arg) == int:
			if dtype is None:
				dtype = float
			self.data = np.empty(arg, dtype=dtype)
		else:
			raise Exception('wrong argument: %s' % type(arg).__name__)

		self.dtype = dtype
		self.shape = self.data.shape


# apply func for arrays or sequences of arrays
def _unwrap(func, *args, opt=()):
	if isinstance(args[0], Array):
		func(*args, *opt)
	else:
		for a in zip(*args):
			func(*a, *opt)


# copy one array to another
def _copyto(dst, src):
	np.copyto(dst.data, src.data)


def copyto(dst, src):
	_unwrap(_copyto, dst, src)


# add two arrays and write to dst
def _add(dst, one, two):
	np.add(one.data, two.data, out=dst.data)


def add(dst, one, two):
	_unwrap(_add, dst, one, two)


# add one array to another
def _addto(dst, src):
	dst.data += src.data


def addto(dst, src):
	_unwrap(_addto, dst, src)


# clip array
def _clip(arr, lv, rv):
	np.clip(arr.data, lv, rv, out=arr.data)


def clip(arr, lv, rv):
	_unwrap(_clip, arr, opt=(lv, rv))

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

	print('add ... ', end='')
	a, b, c = Array(4), Array(4), Array(4)
	a.data[0] = 1
	b.data[0] = 2
	add(c, b, a)
	assert(c.data[0] == 3)
	print('ok')

	print('addto ... ', end='')
	a = Array(4),
	a[0].data[0] = 1
	b = Array(4),
	b[0].data[0] = 2
	addto(b, a)
	assert(b[0].data[0] == 3)
	print('ok')
