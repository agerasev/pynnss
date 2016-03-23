#!/usr/bin/python3

# Path connects source and destination

class Path:
	def __init__(self, src, dst, state=None):
		self.src = src
		self.dst = dst
		self.state = state

class Pipe:
	def __init__(self, data=None):
		self.data = data
