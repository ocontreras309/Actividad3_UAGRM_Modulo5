# Algoritmo genético sin paralelismo ni optimizaciones

import random
import time
import numpy as np

from util import MonitoringThread, show_chessboard

BOARD_SIZE = 8

def generate_population(n_individuals, n_genes):
    indices = np.arange(n_genes)
    population = np.zeros((n_individuals, n_genes), dtype=np.float64)
    for i in range(n_individuals):
        np.random.shuffle(indices)
        population[i] = indices.copy()

    return population

def fitness(individual):
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

def select_individuals(population, fitnesses, probability):
    max_index = int(probability * len(population))
    indices = np.argsort(-fitnesses)
    selected_indices = indices[:max_index]
    selected = np.zeros((max_index, population.shape[1]), dtype=np.float64)
    for i in range(max_index):
        selected[i] = population[selected_indices[i]]
    return selected

def crossover(individual1, individual2):
    crossover_point = random.randint(1, len(individual1)-1)
    result = np.zeros_like(individual1)
    result[:crossover_point] = individual1[:crossover_point]
    result[crossover_point:] = individual2[crossover_point:]
    return result

def mutate(individual, mutation_rate):
    result = individual.copy()
    for i in range(len(result)):
        if random.random() < mutation_rate:
            result[i] = random.randint(0, BOARD_SIZE - 1)
    return result

def train_ga(population_size=1000, iterations=1000):
    population = generate_population(population_size, BOARD_SIZE)
    fitness_history = np.zeros(iterations, dtype=np.float64)
    best_found = np.zeros(BOARD_SIZE, dtype=np.float64)
    
    for i in range(iterations):
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
        
        # Cruzamiento
        offspring = np.zeros((n_pairs, BOARD_SIZE), dtype=np.float64)
        for j in range(n_pairs):
            idx1 = random.randint(0, n_parents-1)
            idx2 = random.randint(0, n_parents-1)
            offspring[j] = crossover(population[idx1], population[idx2])
          
        new_pop_size = n_parents + n_pairs
        population = np.zeros((new_pop_size, BOARD_SIZE), dtype=np.float64)
        population[:n_parents] = population
        population[n_parents:] = offspring
    
        for j in range(len(population)):
            population[j] = mutate(population[j], 0.02)
            
    return best_found, fitness_history


if __name__ == '__main__':
    start = time.time()
    t = MonitoringThread()
    t.start()
    solution, history = train_ga(10000, 1000)
    print("Solution found:", solution)
    print("Fitness history length:", len(history))
    end = time.time()
    print(end - start)
    t.stop()
    
    if len(solution) > 0:
        show_chessboard(solution)
    