import torch
from torch.nn.parameter import Parameter
import operator as op
from functools import reduce
import numpy as np
import itertools as it

def perm(dim_in, num_inputs):
  return np.array(list(it.permutations(range(dim_in)))).T[:num_inputs,:]

def comb(dim_in, num_inputs):
	return np.array(list(list(it.combinations(range(dim_in),num_inputs)))).T

def nCr(n, r):
	r = min(r, n-r)
	numer = reduce(op.mul, range(n, n-r, -1), 1)
	denom = reduce(op.mul, range(1, r+1), 1)
	return int(numer / denom)

def nPr(n, r):
	r = min(r, n-r)
	return int(reduce(op.mul, range(n, n-r, -1), 1))

class AtomicFunction(torch.nn.Module):
	# TODO: add unit bias
	def __init__(self, func, dim_in, num_inputs, commutative=True):
		super(AtomicFunction, self).__init__()
		self.func = func
		self.dim_in = int(dim_in)
		self.num_inputs = int(num_inputs)
		self.commutative = commutative
		self.num_intermediate = None
		self.perm_mat = None
		if self.commutative:
			self.num_intermediate = nCr(self.dim_in, self.num_inputs)
			self.perm_mat = torch.Tensor(comb(self.dim_in, self.num_inputs)).view(-1).long()
		else:
			self.num_intermediate = nPr(self.dim_in, self.num_inputs)
			self.perm_mat = torch.Tensor(perm(self.dim_in, self.num_inputs)).view(-1).long()
		self.weight_inputs = Parameter(torch.rand(1, self.num_inputs, self.num_intermediate))
		self.weight_outputs = Parameter(torch.rand(1, 1, self.num_intermediate))

	def select_inputs(self, x):
		# The input has dimension: N x dim_in
		# We want to select num_inputs of these dimensions at a time, for a total of num_intermediate times
		# That is, if we have inputs a,b,c; then we wan [a,b] then [a,c] then [b,c]
		# This results in an output dimension: N x num_inputs x num_intermediate
		inputs = torch.index_select(x,1,self.perm_mat)
		inputs = inputs.view(inputs.shape[0], self.num_inputs, self.num_intermediate)
		return inputs

	def forward(self, x):
		weighted_inputs = self.weight_inputs * self.select_inputs(x)
		reshaped = weighted_inputs.permute(0,2,1).view(weighted_inputs.shape[0]*self.num_intermediate, self.num_inputs)
		reshaped_outputs = self.func(reshaped)
		outputs = reshaped_outputs.view(weighted_inputs.shape[0], self.num_intermediate, 1)
		combined = torch.sum(torch.matmul(self.weight_inputs, outputs), dim=1)
		return combined

