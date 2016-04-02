#!/usr/bin/python3

import numpy as np


def dcopy(data):
	if data is None:
		return None
	if type(data) == np.ndarray:
		return np.copy(data)
	else:
		lst = []
		for d in data:
			lst.append(np.copy(d))
		return tuple(lst)


def dcopyto(dst, src):
	if dst is not None and src is not None:
		if type(src) == np.ndarray:
			np.copyto(dst, src)
		else:
			for s, d in zip(src, dst):
				np.copyto(d, s)


# unittest
if __name__ == '__main__':
	a = np.array([1, 2, 3, 4])

	print('dcopy ... ', end='')
	b = dcopy(a)
	assert(np.all(a == b))
	b[0] = 10
	assert(a[0] == 1)
	print('ok')

	print('dcopyto ... ', end='')
	b = np.array([0, 0, 0, 0])
	dcopyto(b, a)
	assert(np.all(a == b))
	b[0] = 10
	assert(a[0] == 1)
	print('ok')
