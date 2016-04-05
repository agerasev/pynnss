#!/usr/bin/python3

import numpy as np


class Array:
	def __init__(self, arg, dtype=None, gpu=False):
		self.gpu = gpu

		if type(arg) == Array:
			if dtype is None:
				dtype = arg.dtype
			self.np = np.array(arg.np, dtype=dtype)
		elif type(arg) == np.ndarray:
			if dtype is None:
				dtype = arg.dtype
			self.np = np.array(arg, dtype=dtype)
		elif type(arg) == tuple or type(arg) == int:
			if dtype is None:
				dtype = float
			self.np = np.empty(arg, dtype=dtype)
		else:
			raise Exception('wrong argument: %s' % type(arg).__name__)

		self.dtype = dtype
		self.shape = self.np.shape

	def get(self):
		return np.copy(self.np)

	def set(self, data):
		np.copyto(self.np, data)


def copy(dst, src):
	np.copyto(dst.np, src.np)


def add(dst, one, two):
	np.add(one.np, two.np, out=dst.np)


def radd(dst, arr):
	dst.np += arr.np


def clip(dst, src, lv, rv):
	np.clip(src.np, lv, rv, out=dst.np)


def rclip(dst, lv, rv):
	np.clip(dst.np, lv, rv, out=dst.np)


def mul(dst, one, two):
	if isinstance(two, Array):
		np.mul(one.np, two.np, out=dst.np)
	else:
		np.mul(one.np, two, out=dst.np)


def rmul(dst, src):
	if isinstance(src, Array):
		dst.np *= src.np
	else:
		dst.np *= src


def dot(dst, one, two):
	np.dot(one.np, two.np, out=dst.np)


def raddouter(dst, one, two):
	dst.np += np.outer(one.np, two.np)


def rsubmul(dst, one, two):
	dst.np -= one.np*two


def tanh(dst, src):
	np.tanh(src.np, out=dst.np)


def _f_bptanh(err, out):
	return err*(1 - out**2)

_vf_bptanh = np.vectorize(_f_bptanh)


def bptanh(dst, err, out):
	np.copyto(dst.np, _vf_bptanh(err.np, out.np))
