#!/usr/bin/python3

import numpy as np


class Array:
	def __init__(self, arg, dtype=float, gpu=False):
		self.gpu = gpu
		self.dtype = dtype

		if gpu:
			pass
		else:
			if type(arg) == np.ndarray:
				self.data = arg
			elif type(arg) == tuple or type(arg) == int:
				self.data = np.zeros(arg, dtype=dtype)
			else:
				raise Exception('wrong argument')
			self.size = self.data.shape


# apply func for arrays or sequences of arrays
def _unwrap(func, *args, opt=(), ret=False):
	if isinstance(args[0], Array):
		return func(*args, *opt)
	else:
		if ret:
			lst = []
			for a in zip(*args):
				lst.append(func(*a, *opt))
			return tuple(lst)
		else:
			for a in zip(*args):
				func(*a, *opt)


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
	return _unwrap(_copy, arr, ret=True)


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
