#!/usr/bin/python3

# Path connects source and destination

class Path:
	def __init__(self, src, dst):
		self.src = src
		self.dst = dst


from collections import deque

class Pipe:
	def __init__(self, pipe):
		if pipe is None:
			self.queue = deque()
		else:
			self.queue = deque(pipe.queue)

	def push(self, v):
		self.queue.append(v)

	def pop(self):
		return self.queue.popleft()

	def size(self):
		return len(self.queue)

	def __len__(self):
		return self.size