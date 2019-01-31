from cec2005real.cec2005 import Function
import numpy as np
import functools
import matplotlib.pyplot as plt
from numpy.random import rand
import random
import sys

#benchmark functions
bench_funcs = {
	'sphere' : 1, 		# shifted sphere
	'rastringen' : 9,	# rastringin's function
	'ackley' : 8,		# shifted rotated Ackley's function with global optimum on bounds
	'griewank' : 13,	# expanded extended Griewank's plus Rosenbrock's function
	'schwefel' : 5		# schwefel's problem
}

def rand_mutate(child, info):
	"""
	random;y mutates genes
	"""
	child_len = len(child)
	num_mutations = int(random.uniform(0, child_len))
	mutations = np.random.randint(child_len, size=num_mutations)
	for i in mutations:
		child[i] = random.uniform(info['lower'], info['upper'])
	return child

def swap_mutate(child, info):
	"""
	randomly swaps the alleles of genes
	"""
	child_len = len(child)
	num_swaps = int(random.uniform(0, child_len / 2))
	mutations = np.random.randint(child_len, size=num_swaps)
	if (len(mutations) % 2 != 0):
		mutations = np.delete(mutations, -1)
	for i in range(0, len(mutations), 2):
		loc1 = mutations[i]
		loc2 = mutations[i+1]
		child[loc1], child[loc2] = child[loc2], child[loc1]
	return child

def scramble_mutate(child, info):
	"""
	scrambles sections of a chromosome (child)
	"""
	child_len = len(child)
	point1 = int(random.uniform(0, (child_len / 2) - 1))
	point2 = int(random.uniform(child_len / 2, child_len))
	to_scramble = child[point1:point2]
	random.shuffle(to_scramble)
	child[point1:point2] = to_scramble
	return child

def mutate(mut_func, child, info, chance_of_mutation=0.1):
	"""
	mutates a child/chromosome
	chance_of_muation - the chance that the child will have a mutation
	"""
	if (random.uniform(0, 1) < chance_of_mutation):
		child = mutations[mut_func](child, info)
	return child

#mutation operators
mutations = {
	0 : rand_mutate,
	1 : swap_mutate,
	2 : scramble_mutate
}



def generate_solution(info):
	"""
	generates a random solution which will be a part of the first population
	"""
	dim = info['dimension']
	sol = info['lower'] + rand(dim) * (info['upper'] - info['lower'])
	return sol

def first_population(size, info):
	"""
	returns the first population of randomly generated solutions
	"""
	population = []
	for i in range(size):
		population.append(generate_solution(info))
	return population



def battle(population, pop_size, fitness_func):
	"""
	randomly selects two individuals within the population and returns
	the winner (the one with the higher fitness)
	"""
	opponents = random.sample(list(population), 2)
	individual1 = opponents[0]
	individual2 = opponents[1]
	fitness1 = fitness_func(individual1)
	fitness2 = fitness_func(individual2)
	if (fitness1 < fitness2):
		winner = individual1
	else:
		winner = individual2
	return winner

def tournament(population, fitness_func):
	"""
	returns the pairings of the 'strongest' individuals based on the
	battle results
	"""
	pairings = []
	for _ in range(len(population) // 2):
		sol_len = len(population)
		parent1 = battle(population, sol_len, fitness_func)
		parent2 = battle(population, sol_len, fitness_func)
		pairings.append((parent1, parent2))
	return pairings




def create_child(individual1, individual2, mutation_op, info):
	"""
	creates two children using two-point crossover
	"""
	list1 = list(individual1)
	list2 = list(individual2)
	child1 = []
	child2 = []
	sol_len = len(individual1)
	point1 = int(random.uniform(0, (sol_len / 2) - 1))
	point2 = int(random.uniform(sol_len / 2, sol_len))
	child1 = list1[0:point1] + list2[point1:point2] + list1[point2:sol_len]
	child2 = list2[0:point1] + list1[point1:point2] + list2[point2:sol_len]
	child1 = mutate(mutation_op, child1, info)
	child2 = mutate(mutation_op, child2, info)
	return child1, child2

def create_children(breeders, mutation_op, info):
	"""
	returns a new generation consisting of the children of the 
	previous generation
	"""
	next_gen = []
	for i in range(len(breeders)):
		next_gen.append(create_child(breeders[i][0], breeders[i][1], mutation_op, info))
	return functools.reduce(list.__add__,map(list, next_gen))


# Genetic Algorithm Function
def GA(function, mutation_op, population_size=100, num_gen=50):
	"""
	function - the benchmark function to be run on
	mutation_op - the type of mutation the children will be subjected to
	population_size - number of possible solutions within a generation
	num_gen - number of iterations
	"""
	num_func = bench_funcs[function]
	bench = Function(num_func, 50)
	info = bench.info()
	fitness = bench.get_eval_function()

	population = first_population(population_size, info)
	best = population[0]

	best_of_gen = []

	for _ in range(num_gen):
		for pop in population:
			new_pop_fit = fitness(pop)
			best_fit = fitness(best)
			if (new_pop_fit < best_fit):
				best = pop
				best_fit = new_pop_fit
		breeders = tournament(population, fitness)
		children = create_children(breeders, mutation_op, info)
		population = np.array(children)
		best_of_gen.append(best_fit)

	# each generations best, overall best
	return best_of_gen , best_fit


def avg_std_arr(arr_of_arr):
	"""
	returns the mean and standard deviation of several executions
	"""
	arr_of_arr = list(map(list, zip(*arr_of_arr)))
	avg = []
	std = []
	for arr in arr_of_arr:
		avg.append(np.mean(arr))
		std.append(np.std(arr))
	return avg, std

def err_usage():
	print('usage: GA.py benchmark_function mutation_op')
	print('\nBENCHMARK FUNCTIONS')
	for function in bench_funcs.keys():
		print(function)

	sys.exit()

def norm(i, mini, maxi):
	i = (i - mini) / (maxi - mini)
	return i

#program
def main(argv):
	if (len(sys.argv) != 2):
		err_usage()

	if (not (str(sys.argv[1])) in bench_funcs):
		err_usage()

	bench_func = str(sys.argv[1])
	mutation_op = 2

	res = []
	for i in range(10):
		res.append((GA(bench_func, mutation_op))[0])
		res[i] = [norm(l, min(res[i]), max(res[i])) for l in res[i]]


	avg, std = avg_std_arr(res)
	plt.plot(avg, label='Average')
	plt.plot(std, label='Standard Deviation')
	plt.legend(loc='upper right')
	plt.xlabel('Execution')
	plt.ylabel('Fitness')
	plt.show()
	plt.clf()

if __name__ == "__main__":
	main(sys.argv[1:])
