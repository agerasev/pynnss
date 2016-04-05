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


def copy(dst, src):
	np.copyto(dst.data, src.data)


def add(dst, one, two):
	np.add(one.data, two.data, out=dst.data)


def radd(dst, arr):
	dst.data += arr.data


def clip(dst, src, lv, rv):
	np.clip(src.data, lv, rv, out=dst.data)


def rclip(dst, lv, rv):
	np.clip(dst.data, lv, rv, out=dst.data)


def muls(dst, src, scal):
	np.mul(src.data, scal, out=dst.data)


def mul(dst, one, two):
	np.mul(one.data, two.data, out=dst.data)


def rmuls(dst, scal):
	dst.data *= scal


def rmul(dst, arr):
	np.mul(dst.data, arr.data, out=dst.data)


def tanh(dst, src):
	np.tanh(src, out=dst)


def _f_bptanh(err, out):
	return err*(1 - out**2)

_vf_bptanh = np.vectorize(_f_bptanh)


def bptanh(dst, err, out):
	np.copyto(dst, _vf_bptanh(err, out))
