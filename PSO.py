from cec2005real.cec2005 import Function
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
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

#optimal informant count by function
bench_inf_count = {
	'sphere' : 29, 		
	'rastringen' : 1,	
	'ackley' : 7,		
	'griewank' : 1,	
	'schwefel' : 1		
}

def generate_particle(info):
	"""
	generates a particle whose position values are between the min and 
	max of the benchmark function
	"""
	dim = info['dimension']
	sol = info['lower'] + rand(dim) * (info['upper'] - info['lower'])
	return sol

def generate_swarm(size, info):
	"""
	generates a swarm (group) of particles
	"""
	dim = info['dimension']
	swarm = np.empty((size, dim))
	for i in range(size):
		swarm[i] = generate_particle(info)
	return swarm

def generate_velocities(size, info):
	"""
	returns initial velocities for each particle
	"""
	dim = info['dimension']
	maxi = info['upper']
	mini = info['lower']
	velocities = np.empty((size, dim))
	for i in range(size):
		velocities[i] = np.random.uniform(mini / 20, maxi / 20, dim)
	return velocities

def get_informants(size, inf_count, info):
	"""
	returns inf_count random indices for each particle representing 
	its informants
	"""
	informants = []
	maxi = info['upper']
	informants = np.random.randint(maxi, size=(size, inf_count))
	return informants

def get_best_of_inf(swarm, informants, fitness):
	"""
	rturns the best fitness out of all the informants
	"""
	informants = list(np.array(swarm)[informants])
	fitnesses = list(map(fitness, informants))
	return fitnesses.index(min(fitnesses))

def generate_weights():
	"""
	randomly generates
	alpha = [0, 1]
	beta, gamma, and delta whose values sum up to four, as theoretically, 
	this is the rule of thumb
	"""
	alpha = np.random.uniform(0, 1)
	weights = np.random.dirichlet(np.ones(3), size=4)
	weights = np.transpose(weights)
	weights = [np.sum(w) for w in weights]
	weights.sort()
	delta, gamma, beta = tuple(weights)
	return alpha, beta, gamma, delta

#Particle Swarm Function
def PSO(function, inf_count, swarm_size=100, num_movements=50):
	"""
	function - the benchmark function to be run on
	inf_count - number of informants that each particle has
	swarm_size - number of possible solutions within a generation
	num_movements - the number of times the particles adjust their position
	"""
	num_func = bench_funcs[function]
	bench = Function(num_func, 50)
	info = bench.info()
	fitness = bench.get_eval_function()

	swarm = generate_swarm(swarm_size, info)
	velocities = generate_velocities(swarm_size, info)
	informants = get_informants(swarm_size, inf_count, info)
	alpha, beta, gamma, delta = generate_weights()

	# the index best known position of and individual particle i, init = self
	p_best = np.arange(swarm_size, dtype=int) 
	best_inf_position = np.zeros(swarm_size) # the index of the best known position of an individual i's informants
	g_best = 0 # index of global best location

	# the best fitness calculated after position adjustment
	best_of_movement = []
	for _ in range(num_movements):
		for i in range(len(swarm)):
			particle = swarm[i]
			curr_fit = fitness(particle)
			p_fit = fitness(swarm[p_best[i]])
			if (curr_fit < p_fit):
				p_best[i] = i
			if (fitness(swarm[p_best[i]]) < fitness(swarm[g_best])):
				g_best = p_best[i]
		for i in range(len(swarm)):
			curr_best = swarm[p_best[i]]
			inf_best_index = get_best_of_inf(swarm, informants[i], fitness)
			inf_best = swarm[inf_best_index]
			best_inf_position[i] = inf_best_index
			overall_best = swarm[g_best]
			particle = swarm[i]

			for dim in range(len(particle)):
				b = np.random.uniform(0, beta)
				c = np.random.uniform(0, gamma)
				d = np.random.uniform(0, delta)
				velocities[i] = (alpha * velocities[i]) + (b * (curr_best - particle)) + (c * (inf_best - particle)) + (d * (overall_best - particle))
		for particle, vel in zip(swarm, velocities):
			particle += vel
		best_of_movement.append(fitness(swarm[g_best]))	

	# each adjustments best fitness, overall best fitness
	return best_of_movement, fitness(swarm[g_best])


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
	print('usage: PSO.py benchmark_function')
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

	if (not ((str(sys.argv[1]) in bench_funcs))):
		err_usage()

	bench_func = str(sys.argv[1])
	inf_count = bench_inf_count[bench_func]

	res = []
	for i in range(10):
		res.append((PSO(bench_func, inf_count))[0])
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