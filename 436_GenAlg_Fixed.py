from random import shuffle
from random import uniform
from random import randrange
import numpy as np
import pandas as pd

# NOTE: With the permission of the original author, this code is 
# closely modelled after the code found here: https://desouza.li/genetic-algorithms

def optimize_layout(plants, neighbour_compatibility, population_size=50, num_generations=500, max_mutations=2, num_elite=0):
	"""From a list of plants and a 2D array of weights representing the compatibility of neighbours, use a genetic 
	algorithm to determine a layout that is optimal or very close to optimal"""

	# Convert the compatibility matrix to a Pandas dataframe; label the columns and rows with the plant names
	neighbour_score = pd.DataFrame(data=neighbour_compatibility, index=plants, columns=plants)

	# Generate an initial set of possible garden layouts
	population = get_initial_population(plants, population_size)

	# The incumbent is the current best solution
	incumbent = []
	incumbent_score = 0

	# Total fitness is the fitness of the entire population
	total_fitness = 0
	
	for generation in range(num_generations):
		# Map the layouts in the population to their fitness score
		population_fitness = get_pop_fitness(population, neighbour_score)
		# Sort the population by fitness score
		sorted_population = sort_population(population_fitness)

		# If the best layout in the population has a fitness score greater than the incumbent,
		# Set the incumbent to the new layout and log the change to the console
		if sorted_population.iat[population_size-1,1] > incumbent_score:
			incumbent = sorted_population.iat[population_size-1,0]
			incumbent_score = sorted_population.iat[population_size-1,1]
			print("Generation: %s" %(generation))
			print("Incumbent: %s\nScore:%s" %(incumbent, incumbent_score))

		# Calculate the total fitness of the population
		total_fitness = get_total_fitness(sorted_population)

		new_population = []

		# Keep the best n layouts from the current population (n = num_elite)
		for i in range(num_elite):
			new_population.append(sorted_population.iloc[population_size-(i+1),0])

		# Generate new layouts
		for _ in range(population_size-num_elite):
			# New layouts are crossovers between two layouts
			chromA = get_chromosome(sorted_population, total_fitness)
			chromB = get_chromosome(sorted_population, total_fitness)
			daughter = crossover_chromosome(chromA, chromB)

			# There is a 10% chance that a sequence in the new layout will invert
			if uniform(0,100) > 90:
				daughter = inversion_chromosome(daughter)

			# There is a 25% chance that elements of a layout will exchange positions
			if uniform(0,100) > 75:
				daughter = reciprocal_exchange_chromosome(daughter, max_mutations)

			# Add the new layout to the new population
			new_population.append(daughter)

		population = new_population

	# Log a message to the console with the results
	print("Finished after %s generations!" % (generation))
	print("Plant order: %s" % (incumbent))
	print("Score: %s" % (incumbent_score))

	return incumbent

def get_initial_population(plantIDs, pop_size):
	"""Generate an initial set of layouts from the available plants"""
	population = []
	for _ in range(pop_size):
		# Shuffle because all plants must be in the layout exactly once
		shuffle(plantIDs)
		garden = plantIDs
		population.append(garden)
	return population

def get_fitness(garden, neighbour_score):
	"""Calculate the fitness of a layout by summing the neighbour compatibility score
	for each consecutive pair of plants in the layout"""
	fitness = 0
	for i in range(len(garden)-1):
		fitness += neighbour_score.loc[garden[i], garden[i+1]]
	return fitness

def get_pop_fitness(population, neighbour_score):
	"""Create an array with each layout in a population and its fitness"""
	pop_fitness = [(garden, get_fitness(garden, neighbour_score)) for garden in population]
	population_fitness = pd.DataFrame(pop_fitness)
	return population_fitness

def sort_population(population_fitness):
	"""Sort a population by fitness in ascending order"""
	sorted_population = population_fitness.sort_values(by=1)
	return sorted_population

def get_total_fitness(population_fitness):
	"""Calculate the total fitness of a population as the sum of the 
	fitnesses of each layout in the population"""
	return population_fitness.iloc[:,1].sum()

def get_chromosome(sorted_population, total_fitness):
	"""Pick a chromosome at random from the population;
	chromosomes with larger fitness scores have a greater chance of being selected"""
	random_val = uniform(0, total_fitness)
	cumulative_sum = 0
	for i in range(sorted_population.shape[0]):
		cumulative_sum += sorted_population.iat[i,1]
		if cumulative_sum > random_val:
			return sorted_population.iat[i,0]

def swap(chromosome, pos1, pos2):
	"""Swap the positions of two elements (plants) in a chromosome"""
	plant1 = chromosome[pos1]
	chromosome[pos1] = chromosome[pos2]
	chromosome[pos2] = plant1

def reciprocal_exchange_chromosome(chromosome, max_mutations):
	"""Swap a random number of pairs of elements (plants) in a chromosome"""
	num_mutations = randrange(0,max_mutations)
	for _ in range(num_mutations):
		position1 = randrange(0, len(chromosome))
		position2 = randrange(0, len(chromosome))
		swap(chromosome, position1, position2)
	return chromosome

def inversion_chromosome(chromosome):
	"""Invert a sequence of elements (plants) in a chromosome"""
	pos1 = randrange(0, len(chromosome))
	pos2 = randrange(0, len(chromosome))
	if pos1 > pos2:
		start = pos2
		stop = pos1
	else:
		start = pos1
		stop = pos2
	for i in range((stop-start)//2):
		swap(chromosome, start+i, stop-i)
	return chromosome

def crossover_chromosome(chromA, chromB):
	"""Merge two chromosomes to create a new chromosome"""
	crossover_point = randrange(len(chromA))
	# The first part of the daughter chromosome is the first part of chromosomeA, 
	# up until the crossover_point
	daughter = chromA[:crossover_point]
	# The remainder of the daughter chromosome is the elements from chromosomeB
	# that are not in the daugher chromosome already
	for i in range(len(chromB)):
		if chromB[i] not in daughter:
			daughter.append(chromB[i])
	return daughter