import numpy as np
from copy import deepcopy
from typing import Any
from numpy.random import randint

def TournamentSelect( population, how_many_to_select, tournament_size=4, eps: float = 1e-3, D: np.array = []) -> list:
	# this is a stocastic variation of tournament selection
	pop_size = len(population)
	selection = []

	while len(selection) < how_many_to_select:

		best = population[randint(pop_size)]
		for i in range(tournament_size - 1):
			contestant = population[randint(pop_size)]
			if np.isscalar(best.fitness):
				if contestant.fitness < best.fitness:
					best = contestant
			else:  # when comparing in D dimensions
				best_bigger_cont = contestant.fitness[D] + eps <= best.fitness[D]
				if np.count_nonzero(best_bigger_cont) == len(D):
					best = contestant
		survivor = deepcopy(best)
		selection.append(survivor)

	return selection


def models_selection_wmse(models: list, model_scores: np.array, fitness_function: Any) -> list:
    f_v = list()
    scores_shape = model_scores.shape
    for i in range(scores_shape[1]):
        # because the table is stored as "weights (by rows) x models (by columns)"
        models_for_weight_column = model_scores[:, i]
        best_model_idx = np.argmin(models_for_weight_column)
        model_copy = deepcopy(models[best_model_idx])
        fitness_function.Evaluate(model_copy)
        f_v.append(model_copy)
    return f_v


def models_selection_mse(models: list, top_size: int) -> list:
	top_models = []
	models_fitness_lst = np.array([model.fitness for model in models])
	for model in models:
		if len(top_models) == top_size:
			break
		check_with_other = models_fitness_lst > model.fitness
		if np.sum(check_with_other == True) < top_size:
			top_models.append(model)
	return top_models


def roulette_wheel_selection(population: np.array, choice_size: int) -> np.array:
	indices = range(len(population))
	total_fitness = sum(population)
	probabilities = np.array([p / total_fitness for p in population])
	return np.random.choice(range(len(population)), size=choice_size, p=probabilities)
