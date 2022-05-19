import numpy as np
from copy import deepcopy

class SymbolicRegressionFitness:

	def __init__( self, X_train, y_train, use_weights=False, weights: np.array = np.empty(0)):
		self.X_train = X_train
		self.y_train = y_train
		self.evaluations = 0
		# and deleted some redundant class fields

		self.use_weights = use_weights  # addition for ITGP
		self.weights = weights

	def Evaluate( self, individual):

		self.evaluations = self.evaluations + 1

		output = individual.GetOutput(self.X_train)

		if self.use_weights:  # addition for ITGP
			fit_error = self.weights @ np.square(self.y_train - output)  # wmse case, error in each dim
		else:
			fit_error = np.mean(np.square( self.y_train - output ))  # mse case, one-dim error

		if np.any(np.isnan(fit_error)):
			if np.isscalar(fit_error):
				fit_error = np.inf
			else:
				nan_indices = np.argwhere(np.isnan(fit_error))
				for idx in nan_indices:
					fit_error[idx] = np.inf

		individual.fitness = fit_error  # can be one value or array of values
