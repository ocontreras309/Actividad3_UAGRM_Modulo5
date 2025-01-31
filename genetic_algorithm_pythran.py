# Paralelismo con OpenMP y Pythran

import random
import numpy as np

#pythran export BOARD_SIZE
BOARD_SIZE = 8

#pythran export generate_population(int, int)
def generate_population(n_individuals, n_genes):
    """
    #omp parallel for
    """

    indices = np.arange(n_genes)
    population = np.zeros((n_individuals, n_genes), dtype=np.float64)
    for i in range(n_individuals):
        np.random.shuffle(indices)
        population[i] = indices.copy()

    return population

#pythran export fitness(float64[])
def fitness(individual):
    """
    #omp declare simd
    """
    fitness_value = len(individual)
    for i in range(len(individual)):
        row1 = individual[i]
        for k in range(i + 1, len(individual)):
            row2 = individual[k]
            row_diff = abs(row1 - row2)
            col_diff = abs(i - k)
            if (row_diff == col_diff) or row1 == row2 or i == k:
                fitness_value -= 1
            if fitness_value == 0:
                return 0

    return fitness_value

#pythran export select_individuals(float64[:,:], float64[], float)
def select_individuals(population, fitnesses, probability):
    max_index = int(probability * len(population))
    indices = np.argsort(-fitnesses)  # Returns indices in descending order
    selected_indices = indices[:max_index]
    selected = np.zeros((max_index, population.shape[1]), dtype=np.float64)
    for i in range(max_index):
        selected[i] = population[selected_indices[i]]
    return selected

#pythran export crossover(float64[], float64[])
def crossover(individual1, individual2):
    crossover_point = random.randint(1, len(individual1)-1)
    result = np.zeros_like(individual1)
    result[:crossover_point] = individual1[:crossover_point]
    result[crossover_point:] = individual2[crossover_point:]
    return result

#pythran export mutate(float64[], float)
def mutate(individual, mutation_rate):
    """
    #omp parallel for
    """
    result = individual.copy()
    for i in range(len(result)):
        if random.random() < mutation_rate:
            result[i] = random.randint(0, BOARD_SIZE - 1)
    return result

#pythran export train_ga(int, int)
def train_ga(population_size=1000, iterations=1000):
    """
    Entrenamiento del GA con paralelismo
    """
    population = generate_population(population_size, BOARD_SIZE)
    fitness_history = np.zeros(iterations, dtype=np.float64)
    best_found = np.zeros(BOARD_SIZE, dtype=np.float64)

    for i in range(iterations):
        """
        #omp parallel for
        """
        fitnesses = np.zeros(len(population), dtype=np.float64)
        for j in range(len(population)):
            fitnesses[j] = fitness(population[j])
        
        if len(population) > 0:
            mean_fitness = np.mean(fitnesses)
            fitness_history[i] = mean_fitness
            
            max_fitness = np.max(fitnesses)
            if max_fitness == BOARD_SIZE:
                for j in range(len(population)):
                    if fitnesses[j] == BOARD_SIZE:
                        best_found = population[j].copy()
                        return best_found, fitness_history[:i+1]
        else:
            return best_found, fitness_history[:i+1]

        population = select_individuals(population, fitnesses, 0.65)
        n_parents = len(population)
        n_pairs = n_parents // 2
        
        offspring = np.zeros((n_pairs, BOARD_SIZE), dtype=np.float64)
        for j in range(n_pairs):
            idx1 = random.randint(0, n_parents-1)
            idx2 = random.randint(0, n_parents-1)
            offspring[j] = crossover(population[idx1], population[idx2])
            
        new_pop_size = n_parents + n_pairs
        population = np.zeros((new_pop_size, BOARD_SIZE), dtype=np.float64)
        population[:n_parents] = population
        population[n_parents:] = offspring

        """
        #omp parallel for
        """
        for j in range(len(population)):
            population[j] = mutate(population[j], 0.02)
            
    return best_found, fitness_history