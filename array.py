#!/usr/bin/python3

import numpy as np


class _Array:
	def __init__(self, shape, dtype):
		self.shape = shape
		self.dtype = dtype

	def get(self):
		raise NotImplementedError()

	def set(self, data):
		raise NotImplementedError()


class _ArrayCPU(_Array):
	def __init__(self, nparray):
		_Array.__init__(self, nparray.shape, nparray.dtype)
		self.np = nparray

	def get(self):
		return np.copy(self.np)

	def set(self, data):
		np.copyto(self.np, data)


class _Factory:
	def __init__(self, dtype=np.float64):
		self.dtype = dtype


class _FactoryCPU(_Factory):
	def __init__(self, dtype=np.float64):
		_Factory.__init__(self, dtype)

	def empty(self, shape):
		return _ArrayCPU(np.empty(shape, dtype=self.dtype))

	def zeros(self, shape):
		return _ArrayCPU(np.zeros(shape, dtype=self.dtype))

	def copy(self, array):
		return _ArrayCPU(np.array(array.np, dtype=self.dtype))

	def copynp(self, nparray):
		return _ArrayCPU(np.array(nparray, dtype=self.dtype))


def newFactory(dtype=None, gpu=False):
	return _FactoryCPU(dtype=(np.float64 if dtype is None else dtype))


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
	if isinstance(two, _Array):
		np.mul(one.np, two.np, out=dst.np)
	else:
		np.mul(one.np, two, out=dst.np)


def rmuls(dst, src):
	dst.np *= src


def rmul(dst, src):
	dst.np *= src.np


def dot(dst, one, two):
	np.dot(one.np, two.np, out=dst.np)


def raddouter(dst, one, two):
	dst.np += np.outer(one.np, two.np)


def rsubmuls(dst, one, two):
	dst.np -= one.np*two


def rsubmul(dst, one, two):
	dst.np -= one.np*two.np


def tanh(dst, src):
	np.tanh(src.np, out=dst.np)


def _f_bptanh(err, out):
	return err*(1 - out**2)

_vf_bptanh = np.vectorize(_f_bptanh)


def bptanh(dst, err, out):
	np.copyto(dst.np, _vf_bptanh(err.np, out.np))


def radd_adagrad(dst, grad):
	dst.np += grad.np**2


def _f_adagrad(grad, accum, factor):
	return grad*factor/np.sqrt(accum)

_vf_adagrad = np.vectorize(_f_adagrad)


def rsub_adagrad(dst, grad, factor, rate):
	dst.np -= _vf_adagrad(grad.np, rate.np, factor)
