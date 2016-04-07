#!/usr/bin/python3

import numpy as np
from pynn.node import Node


class Network(Node):
	class _State(Node._State):
		def __init__(self):
			Node._State.__init__(self)
			self.nodes = {}
			self.pipes = []

		def __copy__(self):
			state = Network._State()
			for key in self.nodes:
				state.nodes[key] = copy(self.nodes[key])
			for i in range(len(self.pipes)):
				state.pipes.append(Pipe(self.pipes[i].data))
			return state

	def newState(self):
		state = self._State()
		for key in self.nodes:
			state.nodes[key] = self.nodes[key].newState()
		for i in range(len(self.paths)):
			state.pipes.append(Pipe(self.paths[i].state))
		return state

	class _Error(Node._Error):
		def __init__(self):
			Node._Error.__init__(self)
			self.nodes = {}
			self.pipes = []

		def __copy__(self):
			error = Network._State()
			for key in self.nodes:
				error.nodes[key] = copy(self.nodes[key])
			for i in range(len(self.pipes)):
				error.pipes.append(Pipe(self.pipes[i].data))
			return error

	def newError(self):
		error = self._Error()
		for key in self.nodes:
			error.nodes[key] = self.nodes[key].newError()
		for i in range(len(self.paths)):
			data = None
			if self.paths[i].state is not None:
				data = np.zeros_like(self.paths[i].state)
			error.pipes.append(Pipe(data))
		return error

	class _Gradient(Node._Gradient):
		def __init__(self):
			Node._Gradient.__init__(self)
			self.nodes = {}

		def mul(self, factor):
			for key, node in self.nodes.items():
				if node is not None:
					node.mul(factor)

		def clip(self, value):
			for key, node in self.nodes.items():
				if node is not None:
					node.clip(value)

	def newGradient(self):
		grad = self._Gradient()
		for key in self.nodes:
			grad.nodes[key] = self.nodes[key].newGradient()
		return grad


	def __init__(self, nins, nouts):
		Node.__init__(self, nins, nouts)
		self.ins = [None]*nins
		self.outs = [None]*nouts
		self.nodes = {}
		self.paths = []
		self._flink = {}
		self._blink = {}

	def link(self, path):
		self.paths.append(path)
		self.update()

	def update(self):
		for i in range(len(self.paths)):
			p = self.paths[i]
			self._flink[p.src] = i
			self._blink[p.dst] = i

	class _PropInfo:
		class NodeInfo:
			def __init__(self):
				self.acted = 0
				self.check = 0
		def __init__(self, keys):
			self.nodes = {}
			for key in keys:
				self.nodes[key] = self.NodeInfo()

	# step forward
	def _step(self, state, info):
		count = 0
		for key in list(self.nodes.keys()):
			node = self.nodes[key]

			# check node has not activated yet
			if info.nodes[key].acted != 0:
				# node already activated and has state
				continue

			# check all node input pipes is ready
			pipecount = 0
			for i in range(node.nins):
				pipe = state.pipes[self._blink[(key, i)]]
				if pipe.data is not None:
					pipecount += 1
				else:
					break

			if pipecount != node.nins:
				# node is not ready, check next node
				continue

			info.nodes[key].acted += 1

			# extract inputs from pipes
			vins = []
			for i in range(node.nins):
				pipe = state.pipes[self._blink[(key, i)]]
				vins.append(pipe.data)
				pipe.data = None

			# propagate signal through node
			vouts = node.transmit(state.nodes[key], vins)

			# put outputs into pipes
			for i in range(node.nouts):
				pipe = state.pipes[self._flink[(key, i)]]
				if pipe.data is not None:
					raise Exception('Node ' + str(key) + ' output pipe is not empty')
				pipe.data = vouts[i]

			count += 1
		return count

	# forward propagation
	def _transmit(self, state, vins):
		# prepare propagation info
		info = self._PropInfo(self.nodes.keys())

		# put inputs in pipes
		for i in range(self.nins):
			pidx = self._flink[(-1, i)]
			pipe = state.pipes[pidx]
			if pipe.data is not None:
				raise Exception('Input pipe ' + str(pidx) + ' is not empty')
			pipe.data = vins[i]

		# propagate signals through nodes
		# while there is something to propagate
		count = 1
		while count > 0:
			count = self._step(state, info)

		# extract outputs from pipes
		vouts = []
		for i in range(self.nouts):
			pidx = self._blink[(-1, i)]
			pipe = state.pipes[pidx]
			if pipe.data is None:
				raise Exception('Output pipe ' + str(pidx) + ' is empty')
			vouts.append(pipe.data)
			pipe.data = None

		return vouts


	# step back
	def _backstep(self, grad, error, state, info):
		count = 0
		for key in reversed(list(self.nodes.keys())):
			node = self.nodes[key]

			# check node has not activated yet
			if info.nodes[key].acted != 0:
				# node already activated and has error state
				continue

			# check all node output pipes is ready
			pipecount = 0
			for i in range(node.nouts):
				pipe = error.pipes[self._flink[(key, i)]]
				if pipe.data is not None:
					pipecount += 1
				else:
					break

			if pipecount != node.nouts:
				# node is not ready, check next node
				continue

			info.nodes[key].acted += 1

			# extract ouput errors from pipes
			eouts = []
			for i in range(node.nouts):
				pipe = error.pipes[self._flink[(key, i)]]
				eouts.append(pipe.data)
				pipe.data = None

			# backpropagate error through node
			node_grad = None
			if grad is not None:
				node_grad = grad.nodes[key]
			eins = node.backprop(node_grad, error.nodes[key], state.nodes[key], eouts)

			# put input errors into pipes
			for i in range(node.nins):
				pipe = error.pipes[self._blink[(key, i)]]
				if pipe.data is not None:
					raise Exception('Node ' + str(key) + ' input pipe is not empty')
				pipe.data = eins[i]

			count += 1
		return count

	# error backpropagation
	def _backprop(self, grad, error, state, eouts):
		# prepare propagation info
		info = self._PropInfo(self.nodes.keys())

		# put output errors in pipes
		for i in range(self.nouts):
			pidx = self._blink[(-1, i)]
			pipe = error.pipes[pidx]
			if pipe.data is not None:
				raise Exception('Output pipe ' + str(pidx) + ' is not empty')
			pipe.data = eouts[i]

		# backpropagate errors through nodes
		# while there is something to propagate
		count = 1
		while count > 0:
			count = self._backstep(grad, error, state, info)

		# extract input errors from pipes
		eins = []
		for i in range(self.nins):
			pidx = self._flink[(-1, i)]
			pipe = error.pipes[pidx]
			if pipe.data is None:
				raise Exception('Input pipe ' + str(pidx) + ' is empty')
			eins.append(pipe.data)
			pipe.data = None

		return eins

	# learn network using gradient and learning rate
	def learn(self, grad, rate):
		for key in self.nodes:
			self.nodes[key].learn(grad.nodes[key], rate.nodes[key])
