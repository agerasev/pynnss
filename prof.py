#!/usr/bin/python3

from time import clock


class Empty:
	def __init__(self):
		pass

	def __enter__(self):
		pass

	def __exit__(self, *args):
		pass


class Time(Empty):
	def __init__(self):
		self.start = 0
		self.time = 0

	def __enter__(self):
		self.start = clock()

	def __exit__(self, *args):
		self.time += clock() - self.start
