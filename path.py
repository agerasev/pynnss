#!/usr/bin/python3

# Path connects source and destination of signal

class Path:
	def __init__(self, src, dst):
		assert src.size == dst.size
		self.src = src
		self.dst = dst
		self.stack = []

	def shift(self):
		assert self.dst.data == None
		data = self.src.data
		self.src.data = None
		self.dst.data = data
		self.stack.append(data)

	def clear(self):
		self.stack.clear()
		
	def __repr__(self):
		return '{src: ' + str(self.src) + ', dst: ' + str(self.dst) + ', stack: ' + str(self.stack) + '}'