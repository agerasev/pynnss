#!/usr/bin/python3

# Signal is portion of data NNs operate with

class Signal:
	def __init__(self, data):
		self.data = data
	def __repr__(self):
		return str(self.data)