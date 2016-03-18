#!/usr/bin/python3

from pynn.node import Node
from pynn.path import Pipe

# Network is a Node that contains other Nodes connected with each other with Paths

class Network(Node):
	class State(Node.State):
		def __init__(self):
			Node.State.__init__(self)
			self.nodes = {}
			self.pipes = []

		def __copy__(self):
			state = Network.State()
			for key in self.nodes:
				state.nodes[key] = copy(self.nodes[key])
			for i in range(len(self.pipes)):
				state.pipes.append(Pipe(self.pipes[i].data))
			return state

	def newState(self):
		state = self.State()
		for key in self.nodes:
			state.nodes[key] = self.nodes[key].newState()
		for i in range(len(self.paths)):
			state.pipes.append(Pipe())
		return state

	class Error(Node.Error):
		def __init__(self):
			Node.Error.__init__(self)
			self.nodes = {}
			self.pipes = []

		def __copy__(self):
			error = Network.State()
			for key in self.nodes:
				error.nodes[key] = copy(self.nodes[key])
			for i in range(len(self.pipes)):
				error.pipes.append(Pipe(self.pipes[i].data))
			return error

	def newError(self):
		error = self.Error()
		for key in self.nodes:
			error.nodes[key] = self.nodes[key].newError()
		for i in range(len(self.paths)):
			error.pipes.append(Pipe())
		return error

	class Gradient(Node.Gradient):
		def __init__(self):
			Node.Gradient.__init__(self)
			self.nodes = {}

	def newGradient(self):
		grad = self.Gradient()
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

	def update(self):
		for i in range(len(self.paths)):
			p = self.paths[i]
			self._flink[p.src] = i
			self._blink[p.dst] = i

	class _PropInfo:
		def __init__(self):
			self.activated = {}

	# step forward
	def _step(self, state, info):
		count = 0
		for key in self.nodes:
			node = self.nodes[key]

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

			# extract inputs from pipes
			vins = []
			for i in range(node.nins):
				pipe = state.pipes[self._blink[(key, i)]]
				vins.append(pipe.data)
				pipe.data = None

			# check node has not activated yet
			if info.activated[key] != 0:
				raise Exception('Node ' + str(key) + ' activated twice')
			info.activated[key] += 1

			# propagate signal through node
			vouts = node.feedforward(state.nodes[key], vins)

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
		info = self._PropInfo()
		for key in self.nodes:
			info.activated[key] = 0

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
		for key in self.nodes:
			node = self.nodes[key]

			# check all node output pipes is ready
			pipecount = 0
			for i in range(node.nouts):
				pipe = state.pipes[self._flink[(key, i)]]
				if pipe.data is not None:
					pipecount += 1
				else:
					break

			if pipecount != node.nouts:
				# node is not ready, check next node
				continue

			# extract ouput errors from pipes
			eouts = []
			for i in range(node.nouts):
				pipe = state.pipes[self._flink[(key, i)]]
				eouts.append(pipe.data)
				pipe.data = None

			# check node has not activated yet
			if info.activated[key] != 0:
				raise Exception('Node ' + str(key) + ' activated twice')
			info.activated[key] += 1

			# backpropagate error through node
			eins = node.backprop(grad.nodes[key], error.nodes[key], state.nodes[key], eouts)

			# put input errors into pipes
			for i in range(node.nins):
				pipe = state.pipes[self._blink[(key, i)]]
				if pipe.data is not None:
					raise Exception('Node ' + str(key) + ' input pipe is not empty')
				pipe.data = eins[i]
			count += 1
		return count

	# error backpropagation
	def _backprop(self, grad, error, state, eouts):
		# prepare propagation info
		info = self._PropInfo()
		for key in self.nodes:
			info.activated[key] = 0

		# put output errors in pipes
		for i in range(self.nouts):
			pidx = self._blink[(-1, i)]
			pipe = state.pipes[pidx]
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
			pipe = state.pipes[pidx]
			if pipe.data is None:
				raise Exception('Input pipe ' + str(pidx) + ' is empty')
			eins.append(pipe.data)
			pipe.data = None
		return eins

	# learn network using gradient and learning rate
	def learn(self, grad, rate):
		for key in self.nodes:
			self.nodes[key].learn(grad.nodes[key], rate)
